//! OpenAI 兼容协议适配器（chat 与 chat 上的音频输出）。
//!
//! 这是唯一知道 `/chat/completions` 与相关 wire 结构体的地方，也是
//! wire ↔ 域类型映射的落点。
//!
//! 注意：OpenRouter 的**图片**端点不在这里——它用的是 OpenRouter 自有的
//! `/images` 形状（`resolution` / `aspect_ratio`），属于
//! [`crate::protocols::openrouter`]。

pub mod wire;

use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use std::sync::Arc;
use tracing::warn;

use crate::Message;
use crate::MessageRole;
use crate::capability::{
    ChatCapability, ChatChunk, ChatRequest, GenAudioCapability, GenAudioRequest, GenAudioResponse,
    ModelError,
};
use crate::providers::{Provider, ProviderError};

use wire::{WireAudioConfig, WireChatRequest, WireResponse, WireStreamResponse};

const CHAT_PATH: &str = "/chat/completions";

/// OpenAI 兼容协议适配器。
///
/// 同一个实例可以承担 chat 与音频生成——能力是否真的可用是**端点**的属性，
/// 由注册方断言。
pub struct OpenAiCompatible {
    http: Arc<dyn Provider>,
}

impl OpenAiCompatible {
    pub fn new(http: Arc<dyn Provider>) -> Self {
        Self { http }
    }

    /// 底层传输，便于测试与自定义。
    pub fn http(&self) -> &Arc<dyn Provider> {
        &self.http
    }

    /// 非流式 chat completions。
    pub async fn chat_completions(
        &self,
        wire: &WireChatRequest,
    ) -> Result<WireResponse, ProviderError> {
        let body = serde_json::to_value(wire)?;
        let response = self.http.post_json(CHAT_PATH, body).await?;
        Ok(serde_json::from_value(response)?)
    }

    /// 流式 chat completions。
    ///
    /// SSE 载荷里的 `[DONE]` 与无法解码的事件会被跳过；传输中断会记录告警后
    /// 结束流（`ChatChunk` 本身没有错误位，中断信息通过日志暴露）。
    pub async fn chat_completions_stream(
        &self,
        wire: &WireChatRequest,
    ) -> Result<BoxStream<'static, WireStreamResponse>, ProviderError> {
        let mut wire = wire.clone();
        wire.stream = Some(true);

        let body = serde_json::to_value(&wire)?;
        let mut events = self.http.post_sse(CHAT_PATH, body).await?;

        let decoded = async_stream::stream! {
            while let Some(event) = events.next().await {
                match event {
                    Ok(payload) => {
                        if payload.trim() == "[DONE]" {
                            continue;
                        }
                        match serde_json::from_str::<WireStreamResponse>(&payload) {
                            Ok(parsed) => yield parsed,
                            Err(error) => warn!(
                                %error,
                                payload = %payload,
                                "skipping undecodable stream event"
                            ),
                        }
                    }
                    Err(error) => {
                        warn!(%error, "stream interrupted");
                        break;
                    }
                }
            }
        };

        Ok(Box::pin(decoded))
    }
}

fn message_from_wire(message: wire::WireChoiceMessage) -> Message {
    match message.role {
        MessageRole::Assistant => Message::Assistant {
            content: message.content.unwrap_or_default(),
            reasoning_content: message.reasoning_content,
            tool_calls: message.tool_calls,
        },
        MessageRole::User => Message::User {
            content: message.content.unwrap_or_default(),
        },
        MessageRole::System => Message::System {
            content: message.content.unwrap_or_default(),
        },
        MessageRole::Tool => Message::Tool {
            tool_call_id: message.tool_call_id.unwrap_or_default(),
            content: message.content.unwrap_or_default(),
        },
    }
}

fn chunk_from_wire(response: WireStreamResponse) -> ChatChunk {
    match response.choices.first() {
        Some(choice) => ChatChunk {
            content: choice.delta.content.clone().unwrap_or_default(),
            reasoning_content: choice.delta.reasoning_content.clone().unwrap_or_default(),
            is_finished: choice.finish_reason.is_some(),
            finish_reason: choice.finish_reason.clone(),
            tool_calls: choice.delta.tool_calls.clone(),
            usage: response.usage.map(Into::into),
        },
        None => ChatChunk {
            content: String::new(),
            reasoning_content: String::new(),
            is_finished: true,
            finish_reason: Some("no_choices".to_string()),
            tool_calls: None,
            usage: response.usage.map(Into::into),
        },
    }
}

#[async_trait]
impl ChatCapability for OpenAiCompatible {
    async fn chat(&self, request: ChatRequest) -> Result<Message, ModelError> {
        let wire = WireChatRequest::from(&request);
        let response = self.chat_completions(&wire).await?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or(ModelError::NoResponse)?;

        Ok(message_from_wire(choice.message))
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, ChatChunk>, ModelError> {
        let wire = WireChatRequest::from(&request);
        let stream = self.chat_completions_stream(&wire).await?;

        Ok(stream.map(chunk_from_wire).boxed())
    }
}

#[async_trait]
impl GenAudioCapability for OpenAiCompatible {
    async fn gen_audio(&self, request: GenAudioRequest) -> Result<GenAudioResponse, ModelError> {
        let format = request.format.unwrap_or_else(|| "wav".to_string());
        let base = ChatRequest::new(request.model, vec![Message::user(request.prompt)]);
        let mut wire = WireChatRequest::from(&base);
        wire.modalities = Some(vec!["text".to_string(), "audio".to_string()]);
        wire.audio = Some(WireAudioConfig {
            format: format.clone(),
            voice: request.voice,
        });

        let mut stream = self.chat_completions_stream(&wire).await?;
        let mut audio_data = String::new();
        let mut transcript = String::new();

        while let Some(response) = stream.next().await {
            for choice in response.choices {
                if let Some(audio) = choice.delta.audio {
                    if let Some(data) = audio.data {
                        audio_data.push_str(&data);
                    }
                    if let Some(chunk) = audio.transcript {
                        transcript.push_str(&chunk);
                    }
                }
            }
        }

        if audio_data.is_empty() {
            return Err(ModelError::NoResponse);
        }

        Ok(GenAudioResponse {
            audio_data,
            transcript,
            format,
        })
    }
}

/// 协议层测试共用的假传输：只记录请求、按剧本回放响应。
///
/// 传输缝存在之后，协议层测试不再需要起真实 TCP 服务端。
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use futures::stream;
    use serde_json::Value;
    use std::sync::Mutex;

    #[derive(Default)]
    pub struct FakeTransport {
        requests: Mutex<Vec<(String, Value)>>,
        json: Option<Value>,
        events: Vec<String>,
    }

    impl FakeTransport {
        pub fn with_json(json: Value) -> Arc<Self> {
            Arc::new(Self {
                json: Some(json),
                ..Default::default()
            })
        }

        pub fn with_events(events: Vec<Value>) -> Arc<Self> {
            Self::with_raw_events(events.into_iter().map(|value| value.to_string()).collect())
        }

        pub fn with_raw_events(events: Vec<String>) -> Arc<Self> {
            Arc::new(Self {
                events,
                ..Default::default()
            })
        }

        pub fn recorded(&self) -> Vec<(String, Value)> {
            self.requests.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl Provider for FakeTransport {
        async fn post_json(&self, path: &str, body: Value) -> Result<Value, ProviderError> {
            self.requests.lock().unwrap().push((path.to_string(), body));
            self.json
                .clone()
                .ok_or_else(|| ProviderError::StreamError("no canned json".to_string()))
        }

        async fn post_sse(
            &self,
            path: &str,
            body: Value,
        ) -> Result<BoxStream<'static, Result<String, ProviderError>>, ProviderError> {
            self.requests.lock().unwrap().push((path.to_string(), body));
            Ok(Box::pin(stream::iter(self.events.clone().into_iter().map(Ok))))
        }

        fn name(&self) -> &str {
            "fake"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::FakeTransport;
    use super::*;
    use crate::agent::ToolDef;
    use serde_json::{Value, json};

    fn chat_wire_response(content: &str) -> Value {
        json!({
            "id": "resp_1",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": "stop"
            }]
        })
    }

    fn audio_chunk(data: &str, transcript: &str) -> Value {
        json!({
            "id": "chunk",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "google/lyria-3-clip-preview",
            "system_fingerprint": null,
            "choices": [{
                "index": 0,
                "delta": { "audio": { "data": data, "transcript": transcript } },
                "finish_reason": null
            }]
        })
    }

    #[tokio::test]
    async fn chat_posts_to_chat_completions_and_maps_to_domain_message() {
        let transport = FakeTransport::with_json(chat_wire_response("hello"));
        let model = OpenAiCompatible::new(transport.clone());

        let message = model
            .chat(ChatRequest::new("m", vec![Message::user("hi")]))
            .await
            .unwrap();

        assert_eq!(message, Message::assistant("hello"));
        let requests = transport.recorded();
        assert_eq!(requests[0].0, "/chat/completions");
        assert_eq!(requests[0].1["model"], "m");
        assert!(requests[0].1.get("stream").is_none());
    }

    #[tokio::test]
    async fn chat_stream_decodes_events_and_skips_done_marker() {
        let transport = FakeTransport::with_raw_events(vec![
            json!({
                "id": "1", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "system_fingerprint": null,
                "choices": [{ "index": 0, "delta": { "content": "he" }, "finish_reason": null }]
            })
            .to_string(),
            "[DONE]".to_string(),
            json!({
                "id": "2", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "system_fingerprint": null,
                "choices": [{ "index": 0, "delta": { "content": "llo" }, "finish_reason": "stop" }]
            })
            .to_string(),
        ]);
        let model = OpenAiCompatible::new(transport);

        let chunks = model
            .chat_stream(ChatRequest::new("m", vec![Message::user("hi")]))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].content, "he");
        assert!(!chunks[0].is_finished);
        assert_eq!(chunks[1].content, "llo");
        assert!(chunks[1].is_finished);
        assert_eq!(chunks[1].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn stream_request_forces_stream_flag_at_the_protocol_layer() {
        let transport = FakeTransport::with_events(vec![]);
        let model = OpenAiCompatible::new(transport.clone());

        model
            .chat_stream(ChatRequest::new("m", vec![]))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;

        assert_eq!(transport.recorded()[0].1["stream"], true);
    }

    #[tokio::test]
    async fn gen_audio_builds_modalities_and_collects_streamed_chunks() {
        let transport = FakeTransport::with_events(vec![
            audio_chunk("abc", "hello "),
            audio_chunk("def", "world"),
        ]);
        let model = OpenAiCompatible::new(transport.clone());

        let response = model
            .gen_audio(GenAudioRequest::new(
                "google/lyria-3-clip-preview",
                "Generate a short piano loop",
            ))
            .await
            .unwrap();

        assert_eq!(response.audio_data, "abcdef");
        assert_eq!(response.transcript, "hello world");
        assert_eq!(response.format, "wav");

        let (path, body) = transport.recorded().into_iter().next().unwrap();
        assert_eq!(path, "/chat/completions");
        assert_eq!(body["stream"], true);
        assert_eq!(body["modalities"], json!(["text", "audio"]));
        assert_eq!(body["audio"], json!({ "format": "wav" }));
        assert_eq!(body["messages"][0]["content"], "Generate a short piano loop");
    }

    #[tokio::test]
    async fn tools_are_encoded_into_wire_format() {
        let transport = FakeTransport::with_json(chat_wire_response("ok"));
        let model = OpenAiCompatible::new(transport.clone());

        model
            .chat(
                ChatRequest::new("m", vec![Message::user("hi")]).with_tools(Some(vec![ToolDef {
                    name: "read_file".to_string(),
                    description: "read a file".to_string(),
                    parameters: json!({ "type": "object" }),
                }])),
            )
            .await
            .unwrap();

        let body = &transport.recorded()[0].1;
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(body["tools"][0]["function"]["description"], "read a file");
    }
}
