//! 生成响应、流式事件与流聚合器。
//!
//! 统一承载非流式 [`GenerateResponse`] 与流式 [`GenerateStream`]（[`StreamEvent`] / [`PartDelta`]），
//! 并提供 [`StreamAccumulator`] 与 [`collect_stream`] 供协议层与上层编排共用。

use futures::{StreamExt, stream::BoxStream};
use serde::{Deserialize, Serialize};

use crate::Usage;
use crate::capability::ModelError;
use crate::message::{ContentPart, MediaSource, Message, MessageRole, audio_format_to_mime};

/// 统一的流式生成事件流（显式携带错误）。
pub type GenerateStream = BoxStream<'static, Result<StreamEvent, ModelError>>;

/// 生成结束原因。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Other(String),
}

impl FinishReason {
    pub fn from_wire(raw: &str) -> Self {
        match raw {
            "stop" | "end_turn" | "stop_sequence" | "STOP" => Self::Stop,
            "length" | "max_tokens" | "MAX_TOKENS" => Self::Length,
            "tool_calls" | "tool_use" | "function_call" => Self::ToolCalls,
            "content_filter" | "safety" | "SAFETY" | "refusal" => Self::ContentFilter,
            other => Self::Other(other.to_string()),
        }
    }
}

/// 非流式生成的完整响应。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub message: Message,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// 流式增量内容片段。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PartDelta {
    Text {
        text: String,
    },
    Reasoning {
        text: String,
    },
    ReasoningSignature {
        signature: String,
    },
    Audio {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        transcript: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        format: Option<String>,
    },
}

/// 流式生成的统一事件。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum StreamEvent {
    /// 增量内容片段（文本、推理、推理签名、音频流分片）
    Delta(PartDelta),
    /// 完整产出的内容块（已合并完成的工具调用、图片、文件或带签名的推理块）
    Part(ContentPart),
    /// Token 用量信息（可能出现在 `Finished` 之后）
    Usage(Usage),
    /// 生成结束原因
    Finished { reason: Option<FinishReason> },
}

/// 把一串 [`StreamEvent`] 归并成完整的 [`GenerateResponse`]。
///
/// 供协议层（如必须走流的音频生成）与上层编排共用，避免重复拼接逻辑。
#[derive(Debug, Default)]
pub struct StreamAccumulator {
    parts: Vec<ContentPart>,
    finish_reason: Option<FinishReason>,
    usage: Option<Usage>,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, event: &StreamEvent) {
        match event {
            StreamEvent::Delta(delta) => self.push_delta(delta),
            StreamEvent::Part(part) => self.parts.push(part.clone()),
            StreamEvent::Usage(usage) => self.usage = Some(*usage),
            StreamEvent::Finished { reason } => {
                if reason.is_some() {
                    self.finish_reason = reason.clone();
                }
            }
        }
    }

    fn push_delta(&mut self, delta: &PartDelta) {
        match delta {
            PartDelta::Text { text } => {
                if text.is_empty() {
                    return;
                }
                if let Some(ContentPart::Text { text: existing }) = self.parts.last_mut() {
                    existing.push_str(text);
                } else {
                    self.parts.push(ContentPart::Text { text: text.clone() });
                }
            }
            PartDelta::Reasoning { text } => {
                if text.is_empty() {
                    return;
                }
                if let Some(ContentPart::Reasoning {
                    text: existing,
                    redacted: false,
                    ..
                }) = self.parts.last_mut()
                {
                    existing.push_str(text);
                } else {
                    self.parts.push(ContentPart::reasoning(text.clone()));
                }
            }
            PartDelta::ReasoningSignature { signature } => {
                if let Some(ContentPart::Reasoning {
                    signature: existing,
                    ..
                }) = self.parts.last_mut()
                {
                    match existing {
                        Some(sig) => sig.push_str(signature),
                        None => *existing = Some(signature.clone()),
                    }
                } else {
                    self.parts.push(ContentPart::Reasoning {
                        text: String::new(),
                        signature: Some(signature.clone()),
                        redacted: false,
                    });
                }
            }
            PartDelta::Audio {
                data,
                transcript,
                format,
            } => {
                let has_data = data.as_deref().is_some_and(|d| !d.is_empty());
                let has_transcript = transcript.as_deref().is_some_and(|t| !t.is_empty());
                if !has_data && !has_transcript {
                    return;
                }

                if let Some(ContentPart::Audio {
                    source: MediaSource::Base64 {
                        data: existing_data,
                        ..
                    },
                    transcript: existing_transcript,
                }) = self.parts.last_mut()
                {
                    if let Some(chunk) = data {
                        existing_data.push_str(chunk);
                    }
                    if let Some(chunk) = transcript {
                        match existing_transcript {
                            Some(t) => t.push_str(chunk),
                            None => *existing_transcript = Some(chunk.clone()),
                        }
                    }
                } else {
                    let media_type = audio_format_to_mime(format.as_deref().unwrap_or("wav"));
                    self.parts.push(ContentPart::Audio {
                        source: MediaSource::base64(media_type, data.clone().unwrap_or_default()),
                        transcript: transcript.clone(),
                    });
                }
            }
        }
    }

    pub fn finish(self) -> GenerateResponse {
        GenerateResponse {
            message: Message::new(MessageRole::Assistant, self.parts),
            finish_reason: self.finish_reason,
            usage: self.usage,
            model: None,
            id: None,
        }
    }
}

/// 消费完整事件流并聚合为一次 [`GenerateResponse`]。
pub async fn collect_stream(mut stream: GenerateStream) -> Result<GenerateResponse, ModelError> {
    let mut acc = StreamAccumulator::new();
    while let Some(item) = stream.next().await {
        let event = item?;
        acc.push(&event);
    }
    Ok(acc.finish())
}
