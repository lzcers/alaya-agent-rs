//! OpenAI 兼容协议适配器（chat 与 chat 上的音频输出）。
//!
//! 这是唯一知道 `/chat/completions` 与相关 wire 结构体的地方，也是
//! wire ↔ 域类型映射的落点。它持有一个 [`Provider`]（厂商身份），
//! 地址与认证由对方负责，本层只管形状。
//!
//! 注意：OpenRouter 的**图片**端点不在这里——它用的是 OpenRouter 自有的
//! `/images` 形状（`resolution` / `aspect_ratio`），属于
//! [`crate::protocols::openrouter`]。

pub mod wire;

use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use tracing::warn;

use crate::Message;
use crate::MessageRole;
use crate::capability::{
    ChatCapability, ChatChunk, ChatRequest, GenAudioCapability, GenAudioRequest, GenAudioResponse,
    ModelError,
};
use crate::message::ToolCall;
use crate::providers::Provider;

use wire::{
    WireAudioConfig, WireChatRequest, WireResponse, WireStreamResponse, WireToolCall,
};

const CHAT_PATH: &str = "/chat/completions";

/// OpenAI 兼容协议适配器。
///
/// 同一个实例可以承担 chat 与音频生成——能力是否真的可用是**端点**的属性，
/// 由注册方断言。
pub struct OpenAiCompatible {
    provider: Provider,
}

impl OpenAiCompatible {
    pub fn new(provider: Provider) -> Self {
        Self { provider }
    }

    pub fn provider(&self) -> &Provider {
        &self.provider
    }

    /// 非流式 chat completions。
    pub async fn chat_completions(
        &self,
        wire: &WireChatRequest,
    ) -> Result<WireResponse, ModelError> {
        Ok(self.provider.post_json(CHAT_PATH, wire).await?)
    }

    /// 流式 chat completions。
    ///
    /// SSE 载荷里的 `[DONE]` 与无法解码的事件会被跳过；传输中断会记录告警后
    /// 结束流（`ChatChunk` 本身没有错误位，中断信息通过日志暴露）。
    pub async fn chat_completions_stream(
        &self,
        wire: &WireChatRequest,
    ) -> Result<BoxStream<'static, WireStreamResponse>, ModelError> {
        let mut wire = wire.clone();
        wire.stream = Some(true);

        let mut events = self.provider.post_sse(CHAT_PATH, &wire).await?;

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
            tool_calls: message
                .tool_calls
                .map(|calls| calls.into_iter().map(WireToolCall::into_domain).collect()),
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

/// 把一条增量并进累积状态。
///
/// OpenAI 的流式 tool call 是按 `index` 分片送达的：首个 chunk 给出 `id` 和函数名，
/// 后续 chunk 只带参数文本片段。这是纯 wire 层的产物（Anthropic 用的是
/// `input_json_delta`），所以归并逻辑留在这里，不往上泄漏。
fn accumulate_tool_calls(accumulated: &mut Vec<WireToolCall>, deltas: &[WireToolCall]) {
    for delta in deltas {
        match accumulated
            .iter_mut()
            .find(|existing| existing.same_call(delta))
        {
            Some(existing) => existing.merge_from(delta),
            None => accumulated.push(delta.clone()),
        }
    }
}

/// 累积状态 → 域快照。空列表表示这个响应没有任何工具调用。
fn accumulated_tool_calls(accumulated: &[WireToolCall]) -> Option<Vec<ToolCall>> {
    if accumulated.is_empty() {
        return None;
    }
    Some(
        accumulated
            .iter()
            .cloned()
            .map(WireToolCall::into_domain)
            .collect(),
    )
}

fn chunk_from_wire(response: WireStreamResponse, accumulated: &mut Vec<WireToolCall>) -> ChatChunk {
    let usage = response.usage.map(Into::into);

    match response.choices.first() {
        Some(choice) => {
            if let Some(deltas) = &choice.delta.tool_calls {
                accumulate_tool_calls(accumulated, deltas);
            }

            ChatChunk {
                content: choice.delta.content.clone().unwrap_or_default(),
                reasoning_content: choice.delta.reasoning_content.clone().unwrap_or_default(),
                is_finished: choice.finish_reason.is_some(),
                finish_reason: choice.finish_reason.clone(),
                tool_calls: accumulated_tool_calls(accumulated),
                usage,
            }
        }
        None => ChatChunk {
            content: String::new(),
            reasoning_content: String::new(),
            is_finished: true,
            finish_reason: Some("no_choices".to_string()),
            tool_calls: accumulated_tool_calls(accumulated),
            usage,
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

        // 累积状态归协议层所有：上层拿到的是完整调用，不是需要自己拼的碎片。
        Ok(stream
            .scan(Vec::<WireToolCall>::new(), |accumulated, response| {
                futures::future::ready(Some(chunk_from_wire(response, accumulated)))
            })
            .boxed())
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
    use crate::providers::Provider;
    use crate::transport::{HttpRequest, HttpResponse, Transport, TransportError};
    use bytes::Bytes;
    use serde_json::Value;
    use std::sync::{Arc, Mutex};

    pub const BASE_URL: &str = "https://fake.test/v1";

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

        /// 记录到的 (url, body) 列表。
        pub fn recorded(&self) -> Vec<(String, Value)> {
            self.requests.lock().unwrap().clone()
        }

        /// 记录到的请求路径（去掉 base_url 前缀）。
        pub fn recorded_paths(&self) -> Vec<String> {
            self.recorded()
                .into_iter()
                .map(|(url, _)| url.trim_start_matches(BASE_URL).to_string())
                .collect()
        }

        /// 记录到的第一个请求体。
        pub fn first_body(&self) -> Value {
            self.recorded()
                .into_iter()
                .next()
                .map(|(_, body)| body)
                .unwrap_or(Value::Null)
        }
    }

    /// 用假传输装出一个厂商端点。
    pub fn fake_provider(transport: Arc<FakeTransport>) -> Provider {
        Provider::new(transport, "fake", BASE_URL).with_bearer("test-key")
    }

    #[async_trait]
    impl Transport for FakeTransport {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
            let body = if request.body.is_empty() {
                Value::Null
            } else {
                serde_json::from_slice(&request.body).unwrap_or(Value::Null)
            };
            self.requests
                .lock()
                .unwrap()
                .push((request.url.clone(), body));

            if let Some(json) = &self.json {
                return Ok(HttpResponse::buffered(200, Bytes::from(json.to_string())));
            }

            let mut body = String::new();
            for event in &self.events {
                body.push_str("data: ");
                body.push_str(event);
                body.push_str("\n\n");
            }
            Ok(HttpResponse::buffered(200, Bytes::from(body)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{FakeTransport, fake_provider};
    use super::*;
    use crate::message::{ToolCall, ToolDef};
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
        let model = OpenAiCompatible::new(fake_provider(transport.clone()));

        let message = model
            .chat(ChatRequest::new("m", vec![Message::user("hi")]))
            .await
            .unwrap();

        assert_eq!(message, Message::assistant("hello"));
        assert_eq!(transport.recorded_paths(), vec!["/chat/completions"]);
        assert_eq!(transport.first_body()["model"], "m");
        assert!(transport.first_body().get("stream").is_none());
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
        let model = OpenAiCompatible::new(fake_provider(transport));

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
        let model = OpenAiCompatible::new(fake_provider(transport.clone()));

        model
            .chat_stream(ChatRequest::new("m", vec![]))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;

        assert_eq!(transport.first_body()["stream"], true);
    }

    #[tokio::test]
    async fn gen_audio_builds_modalities_and_collects_streamed_chunks() {
        let transport = FakeTransport::with_events(vec![
            audio_chunk("abc", "hello "),
            audio_chunk("def", "world"),
        ]);
        let model = OpenAiCompatible::new(fake_provider(transport.clone()));

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

        let body = transport.first_body();
        assert_eq!(transport.recorded_paths(), vec!["/chat/completions"]);
        assert_eq!(body["stream"], true);
        assert_eq!(body["modalities"], json!(["text", "audio"]));
        assert_eq!(body["audio"], json!({ "format": "wav" }));
        assert_eq!(body["messages"][0]["content"], "Generate a short piano loop");
    }

    #[tokio::test]
    async fn streaming_tool_call_fragments_become_one_complete_domain_call() {
        let chunk = |delta: Value, finish: Value| {
            json!({
                "id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
                "system_fingerprint": null,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }]
            })
            .to_string()
        };
        let transport = FakeTransport::with_raw_events(vec![
            chunk(
                json!({ "tool_calls": [{ "index": 0, "id": "call_1", "type": "function",
                        "function": { "name": "read_file", "arguments": "" } }] }),
                Value::Null,
            ),
            chunk(
                json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "{\"path\":" } }] }),
                Value::Null,
            ),
            chunk(
                json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "\"a.txt\"}" } }] }),
                Value::Null,
            ),
            chunk(json!({}), json!("tool_calls")),
        ]);
        let model = OpenAiCompatible::new(fake_provider(transport));

        let chunks = model
            .chat_stream(ChatRequest::new("m", vec![Message::user("hi")]))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;

        // 上层拿到的是累积快照，最后一帧是完整调用——不需要自己拼接。
        let last = chunks.last().unwrap();
        assert!(last.is_finished);
        assert_eq!(
            last.tool_calls,
            Some(vec![ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: json!({ "path": "a.txt" }),
            }])
        );
    }

    #[tokio::test]
    async fn assistant_tool_calls_round_trip_back_into_wire_messages() {
        let transport = FakeTransport::with_json(chat_wire_response("ok"));
        let model = OpenAiCompatible::new(fake_provider(transport.clone()));

        let messages = vec![
            Message::user("hi"),
            Message::Assistant {
                content: String::new(),
                reasoning_content: None,
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    arguments: json!({ "path": "a.txt" }),
                }]),
            },
            Message::Tool {
                tool_call_id: "call_1".to_string(),
                content: "{}".to_string(),
            },
        ];

        model
            .chat(ChatRequest::new("m", messages))
            .await
            .unwrap();

        // 域形状 {id, name, arguments} → wire 嵌套 {type, function:{name, arguments 文本}}
        let body = transport.first_body();
        let call = &body["messages"][1]["tool_calls"][0];
        assert_eq!(call["id"], "call_1");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "read_file");
        assert_eq!(call["function"]["arguments"], "{\"path\":\"a.txt\"}");
        assert!(call.get("name").is_none(), "不应泄漏域侧平铺字段");
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_1");
    }

    #[tokio::test]
    async fn tools_are_encoded_into_wire_format() {
        let transport = FakeTransport::with_json(chat_wire_response("ok"));
        let model = OpenAiCompatible::new(fake_provider(transport.clone()));

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

        let body = transport.first_body();
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(body["tools"][0]["function"]["description"], "read a file");
    }
}
