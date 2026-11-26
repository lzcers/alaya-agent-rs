//! 接入一个新模型厂商需要实现什么 —— 这份测试就是答案。
//!
//! 两条路径都覆盖：
//!   - **兼容 OpenAI**：`endpoints::openai_compatible(name, key, base_url)` 一行，**实现 0 个 trait**
//!   - **自有协议**：实现 **1 个 [`GenerateCapability`] trait**；认证方案由 `Provider` 承担，
//!     **不需要自定义传输层**

use async_trait::async_trait;
use futures::stream::StreamExt;
use reqwest::header::{AUTHORIZATION, HeaderMap};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

use alaya_model::capability::{
    FinishReason, GenerateCapability, GenerateRequest, GenerateResponse, GenerateStream,
    ModelError, ModelProfile, PartDelta, StreamEvent,
};
use alaya_model::endpoints::openai_compatible;
use alaya_model::providers::Provider;
use alaya_model::router::ModelRouter;
use alaya_model::transport::{HttpRequest, HttpResponse, Transport, TransportError};
use alaya_model::{Message, Modality};

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

impl<'a> From<&'a GenerateRequest> for AcmeRequest<'a> {
    fn from(request: &'a GenerateRequest) -> Self {
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

/// 唯一的实现：一个统一的 [`GenerateCapability`] trait。
struct AcmeModel {
    provider: Provider,
}

#[async_trait]
impl GenerateCapability for AcmeModel {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError> {
        let wire = AcmeRequest::from(&request);
        let response: AcmeResponse = self.provider.post_json("/v1/generate", &wire).await?;
        let output = response
            .outputs
            .into_iter()
            .next()
            .ok_or(ModelError::NoResponse)?;

        Ok(GenerateResponse {
            message: Message::assistant(output.text),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            model: Some(request.model),
            id: None,
        })
    }

    async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateStream, ModelError> {
        let wire = AcmeRequest::from(&request);
        let mut events = self.provider.post_sse("/v1/generate:stream", &wire).await?;

        let stream = async_stream::stream! {
            while let Some(item) = events.next().await {
                match item {
                    Ok(ev) => yield Ok(StreamEvent::Delta(PartDelta::Text { text: ev.data })),
                    Err(error) => {
                        yield Err(ModelError::from(error));
                        return;
                    }
                }
            }
            yield Ok(StreamEvent::Finished {
                reason: Some(FinishReason::Stop),
            });
        };

        Ok(Box::pin(stream))
    }
}

#[derive(Debug, Clone)]
struct Seen {
    url: String,
    headers: HeaderMap,
    body: Value,
}

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

    let provider = Provider::new(transport.clone(), "acme", "https://api.acme.test")
        .with_header_auth("x-acme-key", "secret");

    let mut router = ModelRouter::new();
    router.register(
        "acme-large",
        ModelProfile::text(),
        Arc::new(AcmeModel { provider }),
    );

    // 非流式
    let response = router
        .generate(GenerateRequest::new(
            "acme-large",
            vec![Message::user("hi")],
        ))
        .await
        .expect("generate should succeed");
    assert_eq!(response.message, Message::assistant("hello from acme"));

    // 流式
    let events: Vec<StreamEvent> = router
        .generate_stream(GenerateRequest::new(
            "acme-large",
            vec![Message::user("hi")],
        ))
        .await
        .expect("generate_stream should succeed")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        events,
        vec![
            StreamEvent::Delta(PartDelta::Text {
                text: "Hel".to_string()
            }),
            StreamEvent::Delta(PartDelta::Text {
                text: "lo".to_string()
            }),
            StreamEvent::Finished {
                reason: Some(FinishReason::Stop)
            },
        ]
    );

    let seen = transport.seen();
    assert_eq!(seen[0].url, "https://api.acme.test/v1/generate");
    assert_eq!(seen[0].body["engine"], "acme-large");
    assert!(seen[0].body.get("model").is_none());
    assert_eq!(seen[1].url, "https://api.acme.test/v1/generate:stream");
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
    router.register(
        "llama-3.3-70b-versatile",
        ModelProfile::text(),
        model.clone(),
    );
    router.register(
        "some-audio-model",
        ModelProfile::audio_generation(),
        model,
    );

    assert!(router.supports_output("llama-3.3-70b-versatile", Modality::Text));
    assert!(router.supports_output("some-audio-model", Modality::Audio));
    assert!(!router.supports_output("llama-3.3-70b-versatile", Modality::Image));
}
