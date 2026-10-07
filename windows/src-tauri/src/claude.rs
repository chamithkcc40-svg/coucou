// Chat API client — Google Gemini (generativelanguage.googleapis.com): multi-turn
// chat with Google Search grounding, and files sent as inlineData / text parts.
//
// (The file keeps its historical name, claude.rs, so the module wiring and the
// base64 helper Stripe uses stay where they were.)
//
// Everything happens here rather than in the island: the API key never leaves
// the Credential Manager, and file bytes never cross the IPC boundary.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::secrets;

/// Base of the Gemini REST API; the model and method are appended per call.
const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";
/// Credential Manager entry holding the Google AI Studio key.
pub const KEY_NAME: &str = "gemini-api-key";
/// Gemini counts its internal "thinking" against this budget too, so it is
/// set higher than the visible answer usually needs.
const MAX_TOKENS: u32 = 8192;
/// Google Search grounding — the Gemini counterpart of the web_search tool the
/// Anthropic version used. Set to false if your key/tier does not allow it.
const WEB_SEARCH: bool = true;
/// Text and code files are inlined; anything larger is skipped, as on macOS.
const MAX_INLINE_TEXT: u64 = 200_000;
/// Gemini refuses requests over 20 MB, and base64 adds a third on top.
const MAX_INLINE_BYTES: u64 = 14_000_000;

/// The one place the chat model is named. The settings window can override it
/// (Settings → Gemini → Model); this is the default and the fallback.
pub const DEFAULT_MODEL: &str = "gemini-3.8-flash";

const SYSTEM_PROMPT: &str = "You are Mochi, a personal AI assistant living at the top of the user's screen. \
You have web search access and can help with absolutely anything — research, coding, finding places, recommendations, tasks, questions. \
Respond in the user's language. Be thorough and complete — use as much detail as the task requires. \
No markdown formatting (no **, no ##, no bullet dashes). Use plain text with line breaks.";

#[derive(Default)]
pub struct Chat {
    /// Full multi-turn history, in Gemini `contents` form (role user / model).
    messages: Mutex<Vec<Value>>,
}

impl Chat {
    pub fn reset(&self) {
        self.messages.lock().unwrap().clear();
    }

    fn is_empty(&self) -> bool {
        self.messages.lock().unwrap().is_empty()
    }

    fn push(&self, message: Value) {
        self.messages.lock().unwrap().push(message);
    }

    fn pop(&self) {
        self.messages.lock().unwrap().pop();
    }

    fn snapshot(&self) -> Vec<Value> {
        self.messages.lock().unwrap().clone()
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum ChatContext {
    File { name: String, path: String },
    Window { app_name: String, title: String, url: Option<String> },
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatReply {
    pub text: String,
}

/// One chat turn. Returns the assistant's text, or a message the island shows
/// in the note view.
pub async fn send(
    chat: &Chat,
    model: &str,
    query: String,
    context: Option<ChatContext>,
) -> Result<ChatReply, String> {
    let key = secrets::get(KEY_NAME)
        .ok_or_else(|| "Gemini API key missing. Add your Google AI Studio key in Settings → Gemini.".to_string())?;

    let model = sanitize_model(model);
    let mut parts: Vec<Value> = Vec::new();

    // File / window context rides along with the first message only, exactly
    // like ClaudeService.chat() on macOS.
    if chat.is_empty() {
        match &context {
            Some(ChatContext::File { name, path }) => {
                if let Some(part) = file_part(path) {
                    parts.push(part);
                }
                parts.push(json!({ "text": format!("File: {name}") }));
            }
            Some(ChatContext::Window { app_name, title, url }) => {
                let mut text = format!("Context — App: {app_name}, Window: {title}");
                if let Some(url) = url {
                    text.push_str(&format!(", URL: {url}"));
                }
                parts.push(json!({ "text": text }));
            }
            None => {}
        }
    }
    parts.push(json!({ "text": query }));

    chat.push(json!({ "role": "user", "parts": parts }));

    let mut body = json!({
        "systemInstruction": { "parts": [{ "text": SYSTEM_PROMPT }] },
        "contents": chat.snapshot(),
        "generationConfig": { "maxOutputTokens": MAX_TOKENS },
    });
    if WEB_SEARCH {
        body["tools"] = json!([{ "google_search": {} }]);
    }

    let streamed = match call(&key, &model, &body).await {
        Ok(s) => s,
        Err(err) => {
            chat.pop(); // keep the history consistent with what the model saw
            return Err(err);
        }
    };

    let text = streamed.text.trim().to_string();

    if text.is_empty() {
        chat.pop();
        if let Some(reason) = streamed.blocked() {
            return Err(format!("Gemini declined this one ({reason})."));
        }
        if streamed.finish_reason.as_deref() == Some("MAX_TOKENS") {
            return Err("Gemini ran out of room before answering. Try a shorter question.".into());
        }
        return Err("No response text.".into());
    }

    // Gemini calls the assistant role "model".
    chat.push(json!({ "role": "model", "parts": [{ "text": text.clone() }] }));
    Ok(ChatReply { text })
}

/// The model id ends up in the URL path: keep only what a model id can contain.
fn sanitize_model(model: &str) -> String {
    let model = model.trim().trim_start_matches("models/");
    let ok = !model.is_empty()
        && model.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_'));
    if ok { model.to_string() } else { DEFAULT_MODEL.to_string() }
}

/// What came out of one streamed response.
#[derive(Debug, Default)]
struct Streamed {
    text: String,
    finish_reason: Option<String>,
    block_reason: Option<String>,
    /// An `error` object sent inside the stream instead of as an HTTP status.
    error: Option<String>,
}

impl Streamed {
    /// Feeds one SSE line. Only `data:` lines carry anything for us.
    fn feed_line(&mut self, line: &str) {
        let line = line.trim_end_matches(['\r', '\n']);
        let Some(data) = line.strip_prefix("data:") else { return };
        let data = data.trim_start();
        if data.is_empty() || data == "[DONE]" {
            return;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else { return };
        self.feed_chunk(&chunk);
    }

    fn feed_chunk(&mut self, chunk: &Value) {
        if let Some(message) = chunk.get("error").and_then(|e| e.get("message")).and_then(Value::as_str) {
            self.error = Some(message.to_string());
            return;
        }
        if let Some(reason) = chunk
            .get("promptFeedback")
            .and_then(|f| f.get("blockReason"))
            .and_then(Value::as_str)
        {
            self.block_reason = Some(reason.to_string());
        }
        let Some(candidate) = chunk.get("candidates").and_then(|c| c.get(0)) else { return };
        // Text is at candidates[0].content.parts[i].text. Usually there is a
        // single part per chunk, but nothing forbids several; "thought" parts
        // are the model's reasoning summary, not the answer.
        if let Some(parts) = candidate.get("content").and_then(|c| c.get("parts")).and_then(Value::as_array) {
            for part in parts {
                if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    continue;
                }
                if let Some(t) = part.get("text").and_then(Value::as_str) {
                    self.text.push_str(t);
                }
            }
        }
        if let Some(reason) = candidate.get("finishReason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
    }

    /// Why Gemini refused, when it did.
    fn blocked(&self) -> Option<String> {
        if let Some(r) = &self.block_reason {
            return Some(r.to_lowercase().replace('_', " "));
        }
        match self.finish_reason.as_deref() {
            Some(r @ ("SAFETY" | "PROHIBITED_CONTENT" | "BLOCKLIST" | "SPII" | "RECITATION")) => {
                Some(r.to_lowercase().replace('_', " "))
            }
            _ => None,
        }
    }
}

async fn call(key: &str, model: &str, body: &Value) -> Result<Streamed, String> {
    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;

    let url = format!("{API_BASE}/{model}:streamGenerateContent?alt=sse");
    let mut response = client
        .post(url)
        .header("x-goog-api-key", key)
        .header("content-type", "application/json")
        .json(body)
        .send()
        .await
        .map_err(network_error)?;

    let status = response.status();
    if !status.is_success() {
        // Errors come back as a plain JSON body, not as an event stream.
        let text = response.text().await.unwrap_or_default();
        return Err(http_error(status.as_u16(), &text, model));
    }

    // Read the event stream as it arrives. Lines are split on raw bytes so a
    // UTF-8 character cut across two network chunks is never mangled.
    let mut streamed = Streamed::default();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(network_error)? {
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            streamed.feed_line(&String::from_utf8_lossy(&line));
        }
    }
    if !buf.is_empty() {
        streamed.feed_line(&String::from_utf8_lossy(&buf));
    }

    if let Some(err) = streamed.error.take() {
        return Err(format!("Gemini API error: {err}"));
    }
    Ok(streamed)
}

fn network_error(e: reqwest::Error) -> String {
    if e.is_timeout() {
        "Gemini took too long to answer. Try again.".into()
    } else if e.is_connect() {
        "Can't reach Google's servers. Check your internet connection.".into()
    } else {
        format!("Network error: {e}")
    }
}

/// Turns a non-2xx answer into something readable in the island.
fn http_error(status: u16, body: &str, model: &str) -> String {
    let parsed = serde_json::from_str::<Value>(body).ok();
    let error = parsed.as_ref().and_then(|v| v.get("error"));
    let detail = error
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| body.chars().take(200).collect());
    // A wrong key is a 400 INVALID_ARGUMENT with reason API_KEY_INVALID, not a 401.
    let bad_key = body.contains("API_KEY_INVALID") || detail.contains("API key not valid");

    match status {
        401 | 403 => format!(
            "Gemini refused the API key (HTTP {status}). Check your Google AI Studio key in Settings → Gemini. {detail}"
        ),
        400 if bad_key => "Gemini API key is not valid. Paste a Google AI Studio key in Settings → Gemini.".into(),
        404 => format!("Model \"{model}\" not found. Pick another model in Settings → Gemini."),
        429 => "Gemini rate limit reached (HTTP 429). Wait a minute and try again, or check your quota in Google AI Studio.".into(),
        500..=599 => format!("Gemini is having trouble right now (HTTP {status}). Try again in a moment."),
        _ => format!("Gemini API {status}: {detail}"),
    }
}

/// PDF / image → inlineData part (base64), text/code → inline text part.
/// Mirrors readFileAsBlock() in ClaudeService.swift.
fn file_part(path: &str) -> Option<Value> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    let mime_type = match ext.as_str() {
        "pdf" => Some("application/pdf"),
        "jpg" | "jpeg" => Some("image/jpeg"),
        "png" => Some("image/png"),
        "gif" => Some("image/gif"),
        "webp" => Some("image/webp"),
        _ => None,
    };

    let len = std::fs::metadata(path).ok()?.len();

    if let Some(mime) = mime_type {
        if len > MAX_INLINE_BYTES {
            return None;
        }
        let bytes = std::fs::read(path).ok()?;
        return Some(json!({
            "inlineData": { "mimeType": mime, "data": base64(&bytes) },
        }));
    }

    if len > MAX_INLINE_TEXT {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    Some(json!({ "text": format!("File contents:\n{text}") }))
}

/// Small standalone base64 encoder — not worth another dependency.
/// Also used for Stripe's basic auth.
pub(crate) fn base64_for(bytes: &[u8]) -> String {
    base64(bytes)
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{base64, http_error, sanitize_model, Streamed, DEFAULT_MODEL};

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn sse_chunks_are_concatenated() {
        let mut s = Streamed::default();
        s.feed_line(r#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Bon"}]}}]}"#);
        s.feed_line("\r\n");
        s.feed_line(r#"data: {"candidates":[{"content":{"role":"model","parts":[{"text":"jour !"}]},"finishReason":"STOP"}]}"#);
        assert_eq!(s.text, "Bonjour !");
        assert_eq!(s.finish_reason.as_deref(), Some("STOP"));
        assert!(s.blocked().is_none());
    }

    #[test]
    fn thought_parts_are_skipped() {
        let mut s = Streamed::default();
        s.feed_line(r#"data: {"candidates":[{"content":{"parts":[{"text":"hmm","thought":true},{"text":"Answer"}]}}]}"#);
        assert_eq!(s.text, "Answer");
    }

    #[test]
    fn blocked_prompt_is_reported() {
        let mut s = Streamed::default();
        s.feed_line(r#"data: {"promptFeedback":{"blockReason":"SAFETY"}}"#);
        assert_eq!(s.blocked().as_deref(), Some("safety"));
    }

    #[test]
    fn in_stream_error_is_kept() {
        let mut s = Streamed::default();
        s.feed_line(r#"data: {"error":{"code":500,"message":"boom"}}"#);
        assert_eq!(s.error.as_deref(), Some("boom"));
    }

    #[test]
    fn errors_are_readable() {
        let bad_key = r#"{"error":{"code":400,"message":"API key not valid. Please pass a valid API key.","status":"INVALID_ARGUMENT","details":[{"reason":"API_KEY_INVALID"}]}}"#;
        assert!(http_error(400, bad_key, "m").contains("not valid"));
        assert!(http_error(403, "{}", "m").contains("refused the API key"));
        assert!(http_error(429, "{}", "m").contains("rate limit"));
        assert!(http_error(404, "{}", "gemini-x").contains("gemini-x"));
        assert!(http_error(503, "", "m").contains("trouble"));
    }

    #[test]
    fn model_is_sanitized() {
        assert_eq!(sanitize_model("gemini-3.7-flash"), "gemini-3.7-flash");
        assert_eq!(sanitize_model("models/gemini-3.8-flash"), "gemini-3.8-flash");
        assert_eq!(sanitize_model("x/../y?z"), DEFAULT_MODEL);
        assert_eq!(sanitize_model(""), DEFAULT_MODEL);
    }
}
