//! 接入一个新模型厂商需要实现什么 —— 这份测试就是答案。
//!
//! 两条路径都覆盖：
//!   - **兼容 OpenAI**：`endpoints::openai_compatible(name, key, base_url)` 一行，**实现 0 个 trait**
//!   - **自有协议**：实现 **1 个 capability trait**；认证方案由 `Provider` 承担，
//!     **不需要自定义传输层**
//!
//! 全文件只使用 `alaya_agent` 的公开 API，证明厂商适配可以完全写在
//! 使用方自己的 crate 里，不需要改 `capability` / `router` / `agent`，
//! 也不需要给我们提 PR。

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use reqwest::header::{AUTHORIZATION, HeaderMap};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

use alaya_agent::Message;
use alaya_agent::capability::{ChatCapability, ChatChunk, ChatRequest, ModelError};
use alaya_agent::endpoints::openai_compatible;
use alaya_agent::providers::Provider;
use alaya_agent::router::ModelRouter;
use alaya_agent::transport::{HttpRequest, HttpResponse, Transport, TransportError};

// ────────────────────────────────────────────────────────────────────────────
// 场景一：自有协议的厂商
//
// Acme 的 wire 字段与 OpenAI 完全不同（`engine` / `prompt` / `max_new_tokens`），
// 认证用 `x-acme-key`。
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AcmeRequest<'a> {
    engine: &'a str,
    prompt: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    max_new_tokens: Option<u32>,
}

impl<'a> From<&'a ChatRequest> for AcmeRequest<'a> {
    fn from(request: &'a ChatRequest) -> Self {
        Self {
            engine: &request.model,
            prompt: &request.messages,
            max_new_tokens: request.max_tokens,
        }
    }
}

#[derive(Debug, Deserialize)]
struct AcmeResponse {
    outputs: Vec<AcmeOutput>,
}

#[derive(Debug, Deserialize)]
struct AcmeOutput {
    text: String,
}

/// 唯一的实现：一个 capability trait。
///
/// 它持有一个 [`Provider`]（传输 + 地址 + 认证），因此不需要自己碰 HTTP。
struct AcmeChat {
    provider: Provider,
}

#[async_trait]
impl ChatCapability for AcmeChat {
    async fn chat(&self, request: ChatRequest) -> Result<Message, ModelError> {
        let wire = AcmeRequest::from(&request);
        let response: AcmeResponse = self.provider.post_json("/v1/generate", &wire).await?;
        let output = response
            .outputs
            .into_iter()
            .next()
            .ok_or(ModelError::NoResponse)?;

        Ok(Message::assistant(output.text))
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, ChatChunk>, ModelError> {
        let wire = AcmeRequest::from(&request);
        let mut events = self.provider.post_sse("/v1/generate:stream", &wire).await?;

        // Acme 的流式事件是纯文本（不是 JSON），所以这里不需要反序列化。
        let stream = async_stream::stream! {
            while let Some(event) = events.next().await {
                match event {
                    Ok(text) => yield ChatChunk {
                        content: text,
                        reasoning_content: String::new(),
                        is_finished: false,
                        finish_reason: None,
                        tool_calls: None,
                        usage: None,
                    },
                    // ChatChunk 没有错误位：流中断只能靠 finish_reason 传达。
                    Err(error) => {
                        yield ChatChunk {
                            content: String::new(),
                            reasoning_content: String::new(),
                            is_finished: true,
                            finish_reason: Some(format!("stream_error: {error}")),
                            tool_calls: None,
                            usage: None,
                        };
                        break;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

// ────────────────────────────────────────────────────────────────────────────
// 注册：上层（agent / hooks / compress）无需任何改动
// ────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Seen {
    url: String,
    headers: HeaderMap,
    body: Value,
}

/// 测试用的假传输：只记录请求并回放剧本。
///
/// 真实场景直接复用 `transport::HttpTransport`——传输层不需要知道 Acme 的存在。
#[derive(Default)]
struct ScriptedTransport {
    seen: Mutex<Vec<Seen>>,
    json: Option<Value>,
    events: Vec<String>,
}

impl ScriptedTransport {
    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl Transport for ScriptedTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        let body = if request.body.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&request.body).unwrap_or(Value::Null)
        };
        self.seen.lock().unwrap().push(Seen {
            url: request.url.clone(),
            headers: request.headers.clone(),
            body,
        });

        // 按 URL 分流：流式端点回放 SSE，其余回放 JSON 剧本。
        if request.url.ends_with(":stream") {
            let mut sse = String::new();
            for event in &self.events {
                sse.push_str("data: ");
                sse.push_str(event);
                sse.push_str("\n\n");
            }
            return Ok(HttpResponse::buffered(200, sse));
        }

        let json = self.json.clone().unwrap_or(Value::Null);
        Ok(HttpResponse::buffered(200, json.to_string()))
    }
}

#[tokio::test]
async fn a_custom_protocol_vendor_needs_one_capability_trait() {
    let transport = Arc::new(ScriptedTransport {
        json: Some(json!({ "outputs": [{ "text": "hello from acme" }] })),
        events: vec!["Hel".to_string(), "lo".to_string()],
        ..Default::default()
    });

    // 认证方案是端点配置，不需要自定义传输层
    let provider = Provider::new(transport.clone(), "acme", "https://api.acme.test")
        .with_header("x-acme-key", "secret");

    let mut router = ModelRouter::new();
    router.add_chat_model("acme-large", Arc::new(AcmeChat { provider }));

    // 非流式
    let message = router
        .chat(ChatRequest::new("acme-large", vec![Message::user("hi")]))
        .await
        .expect("chat should succeed");
    assert_eq!(message, Message::assistant("hello from acme"));

    // 流式
    let chunks = router
        .chat_stream(ChatRequest::new("acme-large", vec![Message::user("hi")]))
        .await
        .expect("chat_stream should succeed")
        .collect::<Vec<_>>()
        .await;
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].content, "Hel");
    assert_eq!(chunks[1].content, "lo");

    // 厂商的 wire 形状在适配器里翻译，router 与 agent 完全不知道 Acme 的存在
    let seen = transport.seen();
    assert_eq!(seen[0].url, "https://api.acme.test/v1/generate");
    assert_eq!(seen[0].body["engine"], "acme-large");
    assert!(seen[0].body.get("model").is_none(), "不应泄漏 OpenAI 字段名");
    assert_eq!(seen[1].url, "https://api.acme.test/v1/generate:stream");

    // 非 Bearer 认证：只有 Acme 自己的头，没有多余的 Authorization
    assert_eq!(seen[0].headers.get("x-acme-key").unwrap(), "secret");
    assert!(seen[0].headers.get(AUTHORIZATION).is_none());
}

// ────────────────────────────────────────────────────────────────────────────
// 场景二：兼容 OpenAI 的厂商，一行接入，零 trait 实现
// ────────────────────────────────────────────────────────────────────────────

#[test]
fn an_openai_compatible_vendor_needs_zero_trait_implementations() {
    let model = Arc::new(openai_compatible(
        "groq",
        "dummy-key",
        "https://api.groq.com/openai/v1",
    ));

    let mut router = ModelRouter::new();
    router.add_chat_model("llama-3.3-70b-versatile", model.clone());
    router.add_audio_model("some-audio-model", model);

    assert!(router.supports_chat("llama-3.3-70b-versatile"));
    assert!(router.supports_audio("some-audio-model"));
    assert!(!router.supports_image("llama-3.3-70b-versatile"));
}
