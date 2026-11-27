//! OpenAI 兼容协议适配器（覆盖文本对话、推理、工具、图文输入以及 chat 端点上的多模态输出）。
//!
//! 协议本身的数据结构定义在 [`types`]，域 ↔ 协议映射在内部 `convert` 模块，
//! 本文件负责 HTTP/SSE 调用以及为 [`OpenAiCompatible`] 实现 [`GenerateCapability`]。

mod convert;
pub mod types;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;
use tracing::warn;

use crate::capability::{
    FinishReason, GenerateCapability, GenerateRequest, GenerateResponse, GenerateStream,
    ModelError, PartDelta, StreamEvent, collect_stream,
};
use crate::message::{ContentPart, Modality};
use crate::providers::Provider;

use types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, StreamOptions, ToolCall,
};

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

    /// 非流式 `POST /chat/completions`。
    pub async fn chat_completions(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse, ModelError> {
        Ok(self.provider.post_json(CHAT_PATH, request).await?)
    }
}

/// 把一条流式工具调用增量并进累积状态。
fn accumulate_tool_calls(accumulated: &mut Vec<ToolCall>, deltas: &[ToolCall]) {
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
        if request.config.wants_output(Modality::Audio) {
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
            .config
            .audio_options()
            .and_then(|o| o.format.clone())
            .unwrap_or_else(|| "wav".to_string());
        let body = ChatCompletionRequest::try_from(&request)?;
        let response = self.chat_completions(&body).await?;
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or(ModelError::NoResponse)?;

        Ok(GenerateResponse {
            message: convert::choice_message_to_domain(choice.message, &default_audio_format),
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
            .config
            .audio_options()
            .and_then(|o| o.format.clone())
            .unwrap_or_else(|| "wav".to_string());
        let mut body = ChatCompletionRequest::try_from(&request)?;
        body.stream = Some(true);
        let has_custom_stream_options = body
            .extra
            .as_ref()
            .is_some_and(|map| map.contains_key("stream_options"));
        if !has_custom_stream_options {
            body.stream_options = Some(StreamOptions {
                include_usage: true,
            });
        }

        let mut events = self.provider.post_sse(CHAT_PATH, &body).await?;

        let stream = async_stream::stream! {
            let mut accumulated_tools: Vec<ToolCall> = Vec::new();

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

                let parsed = match serde_json::from_str::<ChatCompletionChunk>(payload) {
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
                            yield Ok(StreamEvent::Part(ContentPart::ToolCall(call.into())));
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
                yield Ok(StreamEvent::Part(ContentPart::ToolCall(call.into())));
            }
        };

        Ok(Box::pin(stream))
    }
}
