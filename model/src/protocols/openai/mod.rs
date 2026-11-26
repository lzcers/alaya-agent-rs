//! OpenAI 兼容协议适配器（覆盖文本对话、推理、工具、图文输入以及 chat 端点上的多模态输出）。
//!
//! 这是唯一知道 `/chat/completions` 与相关 wire 结构体的地方，也是
//! wire ↔ 域类型映射的落点。它持有一个 [`Provider`]（厂商身份），
//! 地址与认证由对方负责，本层只管形状。

pub mod wire;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;
use tracing::warn;

use crate::capability::{
    FinishReason, GenerateCapability, GenerateRequest, GenerateResponse, GenerateStream,
    ModelError, PartDelta, StreamEvent, collect_stream,
};
use crate::message::{
    ContentPart, MediaSource, Message, MessageRole, Modality, audio_format_to_mime,
};
use crate::providers::Provider;

use wire::{WireChatRequest, WireResponse, WireStreamResponse, WireToolCall};

const CHAT_PATH: &str = "/chat/completions";

/// OpenAI 兼容协议适配器。
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
}

fn message_from_wire(message: wire::WireChoiceMessage, default_audio_format: &str) -> Message {
    let mut parts = Vec::new();

    if let Some(reasoning) = message.reasoning_content
        && !reasoning.is_empty()
    {
        parts.push(ContentPart::reasoning(reasoning));
    }

    if let Some(content) = message.content
        && !content.is_empty()
    {
        parts.push(ContentPart::text(content));
    }

    if let Some(images) = message.images {
        for img in images {
            if !img.image_url.url.trim().is_empty() {
                parts.push(ContentPart::image_url(img.image_url.url));
            }
        }
    }

    if let Some(audio) = message.audio
        && let Some(data) = audio.data.filter(|d| !d.is_empty())
    {
        let format = audio
            .format
            .as_deref()
            .unwrap_or(default_audio_format);
        parts.push(ContentPart::Audio {
            source: MediaSource::base64(audio_format_to_mime(format), data),
            transcript: audio.transcript,
        });
    }

    if let Some(tool_calls) = message.tool_calls {
        for call in tool_calls {
            parts.push(ContentPart::ToolCall(call.into_domain()));
        }
    }

    if message.role == MessageRole::Tool {
        return Message::tool_result_parts(message.tool_call_id.unwrap_or_default(), parts, false);
    }

    Message::new(message.role, parts)
}

/// 把一条流式工具调用增量并进累积状态。
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

/// 检查 SSE payload 是否为服务端在流内下发的错误信封。
fn extract_stream_error(payload: &str) -> Option<String> {
    let value: Value = serde_json::from_str(payload).ok()?;
    let error = value.get("error")?;
    error["message"]
        .as_str()
        .or_else(|| error.as_str())
        .map(ToString::to_string)
}

#[async_trait]
impl GenerateCapability for OpenAiCompatible {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError> {
        // OpenAI / OpenRouter 的 chat 音频输出端点要求流式接收 audio delta，
        // 这里在协议层内部透明走流并聚合成完整 GenerateResponse。
        if request.output_modalities.contains(&Modality::Audio) {
            let mut response = collect_stream(self.generate_stream(request).await?).await?;
            if response.message.content.is_empty() {
                return Err(ModelError::NoResponse);
            }
            if response.finish_reason.is_none() {
                response.finish_reason = Some(FinishReason::Stop);
            }
            return Ok(response);
        }

        let default_audio_format = request
            .audio_options
            .as_ref()
            .and_then(|o| o.format.clone())
            .unwrap_or_else(|| "wav".to_string());
        let wire = WireChatRequest::try_from(&request)?;
        let response = self.chat_completions(&wire).await?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or(ModelError::NoResponse)?;

        Ok(GenerateResponse {
            message: message_from_wire(choice.message, &default_audio_format),
            finish_reason: choice.finish_reason.as_deref().map(FinishReason::from_wire),
            usage: response.usage.map(Into::into),
            model: (!response.model.is_empty()).then_some(response.model),
            id: (!response.id.is_empty()).then_some(response.id),
        })
    }

    async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateStream, ModelError> {
        let default_audio_format = request
            .audio_options
            .as_ref()
            .and_then(|o| o.format.clone())
            .unwrap_or_else(|| "wav".to_string());
        let mut wire = WireChatRequest::try_from(&request)?;
        wire.stream = Some(true);

        let mut events = self.provider.post_sse(CHAT_PATH, &wire).await?;

        let stream = async_stream::stream! {
            let mut accumulated_tools: Vec<WireToolCall> = Vec::new();

            while let Some(item) = events.next().await {
                let sse_event = match item {
                    Ok(ev) => ev,
                    Err(error) => {
                        yield Err(ModelError::from(error));
                        return;
                    }
                };

                if sse_event.event.as_deref() == Some("error") {
                    let message = extract_stream_error(&sse_event.data)
                        .unwrap_or(sse_event.data);
                    yield Err(ModelError::StreamError(message));
                    return;
                }

                let payload = sse_event.data.trim();
                if payload == "[DONE]" {
                    continue;
                }

                if let Some(error_msg) = extract_stream_error(payload) {
                    yield Err(ModelError::StreamError(error_msg));
                    return;
                }

                let parsed = match serde_json::from_str::<WireStreamResponse>(payload) {
                    Ok(parsed) => parsed,
                    Err(error) => {
                        warn!(%error, payload, "skipping undecodable stream event");
                        continue;
                    }
                };

                if let Some(choice) = parsed.choices.into_iter().next() {
                    if let Some(reasoning) = choice.delta.reasoning_content
                        && !reasoning.is_empty()
                    {
                        yield Ok(StreamEvent::Delta(PartDelta::Reasoning { text: reasoning }));
                    }

                    if let Some(text) = choice.delta.content
                        && !text.is_empty()
                    {
                        yield Ok(StreamEvent::Delta(PartDelta::Text { text }));
                    }

                    if let Some(images) = choice.delta.images {
                        for img in images {
                            if !img.image_url.url.trim().is_empty() {
                                yield Ok(StreamEvent::Part(ContentPart::image_url(img.image_url.url)));
                            }
                        }
                    }

                    if let Some(audio) = choice.delta.audio {
                        yield Ok(StreamEvent::Delta(PartDelta::Audio {
                            data: audio.data,
                            transcript: audio.transcript,
                            format: audio.format.or_else(|| Some(default_audio_format.clone())),
                        }));
                    }

                    if let Some(deltas) = &choice.delta.tool_calls {
                        accumulate_tool_calls(&mut accumulated_tools, deltas);
                    }

                    if let Some(reason) = choice.finish_reason {
                        for call in accumulated_tools.drain(..) {
                            yield Ok(StreamEvent::Part(ContentPart::ToolCall(call.into_domain())));
                        }
                        yield Ok(StreamEvent::Finished {
                            reason: Some(FinishReason::from_wire(&reason)),
                        });
                    }
                }

                if let Some(usage) = parsed.usage {
                    yield Ok(StreamEvent::Usage(usage.into()));
                }
            }

            // 兼容省略 finish_reason 的服务端：流末尾兜底冲刷累积工具调用。
            for call in accumulated_tools.drain(..) {
                yield Ok(StreamEvent::Part(ContentPart::ToolCall(call.into_domain())));
            }
        };

        Ok(Box::pin(stream))
    }
}

/// 协议层测试共用的假传输：只记录请求、按剧本回放响应。
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

        pub fn recorded(&self) -> Vec<(String, Value)> {
            self.requests.lock().unwrap().clone()
        }

        pub fn recorded_paths(&self) -> Vec<String> {
            self.recorded()
                .into_iter()
                .map(|(url, _)| url.trim_start_matches(BASE_URL).to_string())
                .collect()
        }

        pub fn first_body(&self) -> Value {
            self.recorded()
                .into_iter()
                .next()
                .map(|(_, body)| body)
                .unwrap_or(Value::Null)
        }
    }

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
    async fn generate_posts_to_chat_completions_and_maps_to_domain_response() {
        let transport = FakeTransport::with_json(chat_wire_response("hello"));
        let model = OpenAiCompatible::new(fake_provider(transport.clone()));

        let response = model
            .generate(GenerateRequest::new("m", vec![Message::user("hi")]))
            .await
            .unwrap();

        assert_eq!(response.message, Message::assistant("hello"));
        assert_eq!(response.finish_reason, Some(FinishReason::Stop));
        assert_eq!(transport.recorded_paths(), vec!["/chat/completions"]);
        assert_eq!(transport.first_body()["model"], "m");
        assert!(transport.first_body().get("stream").is_none());
    }

    #[tokio::test]
    async fn generate_stream_decodes_events_and_skips_done_marker() {
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

        let events: Vec<StreamEvent> = model
            .generate_stream(GenerateRequest::new("m", vec![Message::user("hi")]))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(
            events,
            vec![
                StreamEvent::Delta(PartDelta::Text {
                    text: "he".to_string()
                }),
                StreamEvent::Delta(PartDelta::Text {
                    text: "llo".to_string()
                }),
                StreamEvent::Finished {
                    reason: Some(FinishReason::Stop)
                },
            ]
        );
    }

    #[tokio::test]
    async fn stream_request_forces_stream_flag_at_the_protocol_layer() {
        let transport = FakeTransport::with_events(vec![]);
        let model = OpenAiCompatible::new(fake_provider(transport.clone()));

        model
            .generate_stream(GenerateRequest::new("m", vec![]))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;

        assert_eq!(transport.first_body()["stream"], true);
    }

    #[tokio::test]
    async fn audio_generate_builds_modalities_and_collects_streamed_chunks() {
        let transport = FakeTransport::with_events(vec![
            audio_chunk("abc", "hello "),
            audio_chunk("def", "world"),
        ]);
        let model = OpenAiCompatible::new(fake_provider(transport.clone()));

        let response = model
            .generate(GenerateRequest::audio(
                "google/lyria-3-clip-preview",
                "Generate a short piano loop",
            ))
            .await
            .unwrap();

        let audios = response.message.audios();
        assert_eq!(audios.len(), 1);
        assert_eq!(
            audios[0].0,
            &MediaSource::base64("audio/wav", "abcdef")
        );
        assert_eq!(audios[0].1, Some("hello world"));

        let body = transport.first_body();
        assert_eq!(transport.recorded_paths(), vec!["/chat/completions"]);
        assert_eq!(body["stream"], true);
        assert_eq!(body["modalities"], json!(["text", "audio"]));
        assert_eq!(body["audio"], json!({ "format": "wav" }));
        assert_eq!(
            body["messages"][0]["content"],
            "Generate a short piano loop"
        );
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

        let response = collect_stream(
            model
                .generate_stream(GenerateRequest::new("m", vec![Message::user("hi")]))
                .await
                .unwrap(),
        )
        .await
        .unwrap();

        assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
        assert_eq!(
            response.message.tool_calls(),
            vec![&ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: json!({ "path": "a.txt" }),
            }]
        );
    }

    #[tokio::test]
    async fn assistant_tool_calls_round_trip_back_into_wire_messages() {
        let transport = FakeTransport::with_json(chat_wire_response("ok"));
        let model = OpenAiCompatible::new(fake_provider(transport.clone()));

        let messages = vec![
            Message::user("hi"),
            Message::assistant_parts(vec![ContentPart::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: json!({ "path": "a.txt" }),
            })]),
            Message::tool_result("call_1", "{}"),
        ];

        model
            .generate(GenerateRequest::new("m", messages))
            .await
            .unwrap();

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
            .generate(
                GenerateRequest::new("m", vec![Message::user("hi")]).with_tools(Some(vec![
                    ToolDef {
                        name: "read_file".to_string(),
                        description: "read a file".to_string(),
                        parameters: json!({ "type": "object" }),
                    },
                ])),
            )
            .await
            .unwrap();

        let body = transport.first_body();
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "read_file");
        assert_eq!(body["tools"][0]["function"]["description"], "read a file");
    }
}
