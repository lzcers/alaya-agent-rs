//! 统一的多模态生成能力接口。
//!
//! 文本对话、视觉理解、语音对话、推理、工具调用、生图、生音等所有生成式模型交互
//! 统一收敛到 [`GenerateCapability`]：多模态 [`Message`] 输入 + 期望输出模态
//! [`Modality`] + 统一的 [`GenerateResponse`] / [`StreamEvent`] 输出。

use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Usage;
use crate::capability::ModelError;
use crate::message::{
    ContentPart, MediaSource, Message, MessageRole, Modality, ToolDef, audio_format_to_mime,
};

/// 统一的流式生成事件流（显式携带错误）。
pub type GenerateStream = BoxStream<'static, Result<StreamEvent, ModelError>>;

/// 推理强度档位。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// 厂商无关的推理配置。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ReasoningConfig {
    /// 使用模型默认策略
    #[default]
    Default,
    /// 显式关闭推理
    Disabled,
    /// 显式开启推理
    Enabled,
    /// 按档位控制推理强度
    Effort { level: ReasoningEffort },
    /// 按 token 预算控制推理深度（Anthropic / Gemini）
    Budget { tokens: u32 },
}

/// 结构化输出格式。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    #[default]
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(default)]
        strict: bool,
    },
}

/// 工具选择策略。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Tool {
        name: String,
    },
}

/// 图片输出参数。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
}

impl ImageOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    pub fn with_count(mut self, count: u32) -> Self {
        self.count = Some(count);
        self
    }
}

/// 音频输出参数。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
}

impl AudioOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }
}

/// 一次统一的多模态生成请求。
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// 逻辑模型名，同时作为分发层路由键。
    pub model: String,
    pub messages: Vec<Message>,
    /// 期望的输出模态，默认 `[Modality::Text]`。
    pub output_modalities: Vec<Modality>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub tools: Option<Vec<ToolDef>>,
    pub tool_choice: Option<ToolChoice>,
    pub response_format: ResponseFormat,
    pub reasoning: ReasoningConfig,
    pub image_options: Option<ImageOptions>,
    pub audio_options: Option<AudioOptions>,
    /// 要求流式响应携带 token 用量。
    pub include_usage: bool,
    /// 厂商私有参数逃生舱（协议层可选择性透传）。
    pub provider_options: Option<Value>,
}

impl GenerateRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            output_modalities: vec![Modality::Text],
            temperature: None,
            top_p: None,
            max_tokens: None,
            stop_sequences: None,
            tools: None,
            tool_choice: None,
            response_format: ResponseFormat::Text,
            reasoning: ReasoningConfig::Default,
            image_options: None,
            audio_options: None,
            include_usage: false,
            provider_options: None,
        }
    }

    /// 快捷构造：纯图片生成请求。
    pub fn image(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self::new(model, vec![Message::user(prompt)]).with_output_modalities([Modality::Image])
    }

    /// 快捷构造：音频生成请求。
    pub fn audio(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self::new(model, vec![Message::user(prompt)]).with_output_modalities([Modality::Audio])
    }

    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn with_output_modalities(
        mut self,
        modalities: impl IntoIterator<Item = Modality>,
    ) -> Self {
        self.output_modalities = modalities.into_iter().collect();
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_stop_sequences(mut self, stops: Vec<String>) -> Self {
        self.stop_sequences = Some(stops);
        self
    }

    pub fn with_tools(mut self, tools: Option<Vec<ToolDef>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = format;
        self
    }

    pub fn with_reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = reasoning;
        self
    }

    pub fn with_image_options(mut self, options: ImageOptions) -> Self {
        self.image_options = Some(options);
        self
    }

    pub fn with_audio_options(mut self, options: AudioOptions) -> Self {
        self.audio_options = Some(options);
        self
    }

    pub fn with_stream_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = include_usage;
        self
    }

    pub fn with_provider_options(mut self, options: Value) -> Self {
        self.provider_options = Some(options);
        self
    }

    /// 汇总请求消息中实际出现的所有输入模态。
    pub fn input_modalities(&self) -> Vec<Modality> {
        let mut out = Vec::new();
        for message in &self.messages {
            for part in &message.content {
                part.collect_modalities(&mut out);
            }
        }
        out
    }
}

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

/// 模型能力画像：声明该模型支持的输入/输出模态与关键能力。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProfile {
    pub input_modalities: Vec<Modality>,
    pub output_modalities: Vec<Modality>,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub supports_streaming: bool,
    pub context_window: Option<u32>,
}

impl ModelProfile {
    /// 纯文本对话模型（支持工具与流式）。
    pub fn text() -> Self {
        Self {
            input_modalities: vec![Modality::Text],
            output_modalities: vec![Modality::Text],
            supports_tools: true,
            supports_reasoning: false,
            supports_streaming: true,
            context_window: None,
        }
    }

    /// 视觉语言模型（文本+图片输入 → 文本输出）。
    pub fn vision() -> Self {
        Self {
            input_modalities: vec![Modality::Text, Modality::Image],
            output_modalities: vec![Modality::Text],
            supports_tools: true,
            supports_reasoning: false,
            supports_streaming: true,
            context_window: None,
        }
    }

    /// 全模态模型（全输入模态 → 文本/图片/音频输出，支持工具与推理）。
    pub fn omni() -> Self {
        Self {
            input_modalities: vec![
                Modality::Text,
                Modality::Image,
                Modality::Audio,
                Modality::Video,
                Modality::File,
            ],
            output_modalities: vec![Modality::Text, Modality::Image, Modality::Audio],
            supports_tools: true,
            supports_reasoning: true,
            supports_streaming: true,
            context_window: None,
        }
    }

    /// 专用生图模型（文本输入 → 图片输出）。
    pub fn image_generation() -> Self {
        Self {
            input_modalities: vec![Modality::Text],
            output_modalities: vec![Modality::Image],
            supports_tools: false,
            supports_reasoning: false,
            supports_streaming: false,
            context_window: None,
        }
    }

    /// 专用音频生成模型（文本输入 → 音频输出）。
    pub fn audio_generation() -> Self {
        Self {
            input_modalities: vec![Modality::Text],
            output_modalities: vec![Modality::Audio],
            supports_tools: false,
            supports_reasoning: false,
            supports_streaming: true,
            context_window: None,
        }
    }

    pub fn with_input_modalities(mut self, modalities: impl IntoIterator<Item = Modality>) -> Self {
        self.input_modalities = modalities.into_iter().collect();
        self
    }

    pub fn with_output_modalities(mut self, modalities: impl IntoIterator<Item = Modality>) -> Self {
        self.output_modalities = modalities.into_iter().collect();
        self
    }

    pub fn with_tools(mut self, supported: bool) -> Self {
        self.supports_tools = supported;
        self
    }

    pub fn with_reasoning(mut self, supported: bool) -> Self {
        self.supports_reasoning = supported;
        self
    }

    pub fn with_streaming(mut self, supported: bool) -> Self {
        self.supports_streaming = supported;
        self
    }

    pub fn with_context_window(mut self, tokens: u32) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// 在调用上游前校验请求模态与工具需求是否满足该模型画像。
    pub fn validate(&self, request: &GenerateRequest) -> Result<(), ModelError> {
        for modality in request.input_modalities() {
            if !self.input_modalities.contains(&modality) {
                return Err(ModelError::Unsupported(format!(
                    "model `{}` does not support `{modality}` input",
                    request.model
                )));
            }
        }
        for modality in &request.output_modalities {
            if !self.output_modalities.contains(modality) {
                return Err(ModelError::Unsupported(format!(
                    "model `{}` does not support `{modality}` output",
                    request.model
                )));
            }
        }
        if request.tools.as_ref().is_some_and(|t| !t.is_empty()) && !self.supports_tools {
            return Err(ModelError::Unsupported(format!(
                "model `{}` does not support tools",
                request.model
            )));
        }
        Ok(())
    }
}

/// 统一的多模态生成能力 trait。
///
/// 由协议适配器（[`crate::protocols`]）与路由器（[`crate::router::ModelRouter`]）实现。
#[async_trait]
pub trait GenerateCapability: Send + Sync {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError>;

    async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateStream, ModelError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_builders_configure_domain_fields() {
        let request = GenerateRequest::new("model", Vec::new())
            .with_output_modalities([Modality::Text, Modality::Image])
            .with_response_format(ResponseFormat::JsonObject)
            .with_reasoning(ReasoningConfig::Effort {
                level: ReasoningEffort::High,
            })
            .with_temperature(0.2)
            .with_max_tokens(512)
            .with_stream_usage(true);

        assert_eq!(
            request.output_modalities,
            vec![Modality::Text, Modality::Image]
        );
        assert_eq!(request.response_format, ResponseFormat::JsonObject);
        assert!(request.include_usage);
        assert_eq!(request.temperature, Some(0.2));
        assert_eq!(request.max_tokens, Some(512));
        assert_eq!(
            request.reasoning,
            ReasoningConfig::Effort {
                level: ReasoningEffort::High
            }
        );
    }

    #[test]
    fn tools_stay_domain_typed() {
        let tool = ToolDef {
            name: "read_file".to_string(),
            description: "read".to_string(),
            parameters: json!({ "type": "object" }),
        };
        let request = GenerateRequest::new("model", Vec::new()).with_tools(Some(vec![tool]));

        assert_eq!(request.tools.unwrap()[0].name, "read_file");
    }
}
