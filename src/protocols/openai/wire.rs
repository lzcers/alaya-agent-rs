//! OpenAI 兼容协议的 wire 结构体。
//!
//! 这里是「线上长什么样」的唯一真相：字段名、别名、容错解析、以及
//! 域请求 ↔ wire body 的双向翻译。域层不应该看见本模块的任何类型。

use serde::de::{self, Unexpected};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use tracing::warn;

use crate::capability::{
    GenerateRequest, ModelError, ReasoningConfig, ResponseFormat, ToolChoice,
};
use crate::message::{
    ContentPart, MediaSource, Message, MessageRole, Modality, ToolCall, mime_to_audio_format,
};

// ============================================================================
// 容错反序列化
// ============================================================================

fn value_to_u32_lossy<E>(value: Value) -> Result<u32, E>
where
    E: de::Error,
{
    match value {
        Value::Number(n) => {
            if let Some(v) = n.as_u64() {
                u32::try_from(v).map_err(|_| E::invalid_value(Unexpected::Unsigned(v), &"u32"))
            } else if let Some(v) = n.as_i64() {
                u32::try_from(v).map_err(|_| E::invalid_value(Unexpected::Signed(v), &"u32"))
            } else if let Some(v) = n.as_f64() {
                if v.is_finite() && v >= 0.0 && v <= u32::MAX as f64 {
                    Ok(v.round() as u32)
                } else {
                    Err(E::invalid_value(
                        Unexpected::Float(v),
                        &"finite non-negative u32",
                    ))
                }
            } else {
                Err(E::custom("invalid number for u32"))
            }
        }
        Value::String(s) => {
            if let Ok(v) = s.parse::<u32>() {
                Ok(v)
            } else if let Ok(v) = s.parse::<f64>() {
                if v.is_finite() && v >= 0.0 && v <= u32::MAX as f64 {
                    Ok(v.round() as u32)
                } else {
                    Err(E::invalid_value(
                        Unexpected::Float(v),
                        &"finite non-negative u32",
                    ))
                }
            } else {
                Err(E::invalid_value(
                    Unexpected::Str(&s),
                    &"u32-compatible number",
                ))
            }
        }
        other => Err(E::custom(format!(
            "expected u32-compatible number, got {}",
            other
        ))),
    }
}

fn deserialize_u32_lossy<'de, D>(deserializer: D) -> Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    value_to_u32_lossy(Value::deserialize(deserializer)?)
}

fn deserialize_option_u32_lossy<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    match Option::<Value>::deserialize(deserializer)? {
        Some(value) => value_to_u32_lossy(value).map(Some),
        None => Ok(None),
    }
}

// ============================================================================
// 请求
// ============================================================================

/// `POST /chat/completions` 的请求体。
#[derive(Debug, Clone, Serialize)]
pub struct WireChatRequest {
    pub model: String,
    pub messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<WireToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<WireResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<WireThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<WireStreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<WireAudioConfig>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireInputAudio {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireInputFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireContentPart {
    Text { text: String },
    ImageUrl { image_url: WireImageUrl },
    InputAudio { input_audio: WireInputAudio },
    File { file: WireInputFile },
}

/// 纯文本直接序列化为字符串，多模态序列化为 OpenAI content parts 数组。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum WireContent {
    Text(String),
    Parts(Vec<WireContentPart>),
}

/// 请求体里的消息。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum WireMessage {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: WireContent },
    #[serde(rename = "assistant")]
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<WireToolCall>>,
    },
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: WireContent,
    },
}

fn wire_content_from_parts(parts: &[ContentPart]) -> Result<WireContent, ModelError> {
    if parts.is_empty() {
        return Ok(WireContent::Text(String::new()));
    }

    if parts
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }))
    {
        if let [ContentPart::Text { text }] = parts {
            return Ok(WireContent::Text(text.clone()));
        }
        let joined = parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(WireContent::Text(joined));
    }

    let mut wire_parts = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            ContentPart::Text { text } => {
                wire_parts.push(WireContentPart::Text { text: text.clone() });
            }
            ContentPart::Image { source, detail } => {
                wire_parts.push(WireContentPart::ImageUrl {
                    image_url: WireImageUrl {
                        url: source.to_uri().into_owned(),
                        detail: detail.clone(),
                    },
                });
            }
            ContentPart::Audio { source, .. } => match source {
                MediaSource::Base64 { media_type, data } => {
                    wire_parts.push(WireContentPart::InputAudio {
                        input_audio: WireInputAudio {
                            data: data.clone(),
                            format: mime_to_audio_format(media_type).to_string(),
                        },
                    });
                }
                _ => {
                    return Err(ModelError::Unsupported(
                        "OpenAI chat completions only supports inline base64 audio input"
                            .to_string(),
                    ));
                }
            },
            ContentPart::File { source, filename } => {
                let file = match source {
                    MediaSource::Base64 { .. } => WireInputFile {
                        filename: filename.clone(),
                        file_data: Some(source.to_uri().into_owned()),
                        file_id: None,
                    },
                    MediaSource::FileRef { uri, .. } => WireInputFile {
                        filename: filename.clone(),
                        file_data: None,
                        file_id: Some(uri.clone()),
                    },
                    MediaSource::Url { .. } => {
                        return Err(ModelError::Unsupported(
                            "OpenAI chat completions file input requires base64 data or file_id"
                                .to_string(),
                        ));
                    }
                };
                wire_parts.push(WireContentPart::File { file });
            }
            ContentPart::Video { .. } => {
                return Err(ModelError::Unsupported(
                    "OpenAI chat completions does not support video input parts".to_string(),
                ));
            }
            ContentPart::Reasoning { .. }
            | ContentPart::ToolCall(_)
            | ContentPart::ToolResult { .. } => {}
        }
    }

    Ok(WireContent::Parts(wire_parts))
}

fn wire_messages_from_domain(messages: &[Message]) -> Result<Vec<WireMessage>, ModelError> {
    let mut out = Vec::with_capacity(messages.len());

    for message in messages {
        match message.role {
            MessageRole::System => {
                out.push(WireMessage::System {
                    content: message.text().into_owned(),
                });
            }
            MessageRole::User => {
                out.push(WireMessage::User {
                    content: wire_content_from_parts(&message.content)?,
                });
            }
            MessageRole::Assistant => {
                let content = message.text().into_owned();
                let reasoning_content = message.reasoning();
                let tool_calls: Vec<WireToolCall> = message
                    .tool_calls()
                    .into_iter()
                    .map(WireToolCall::from)
                    .collect();
                out.push(WireMessage::Assistant {
                    content,
                    reasoning_content,
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
                });
            }
            MessageRole::Tool => {
                let mut emitted = false;
                for part in &message.content {
                    if let ContentPart::ToolResult {
                        tool_call_id,
                        content,
                        ..
                    } = part
                    {
                        out.push(WireMessage::Tool {
                            tool_call_id: tool_call_id.clone(),
                            content: wire_content_from_parts(content)?,
                        });
                        emitted = true;
                    }
                }
                if !emitted {
                    out.push(WireMessage::Tool {
                        tool_call_id: String::new(),
                        content: wire_content_from_parts(&message.content)?,
                    });
                }
            }
        }
    }

    Ok(out)
}

/// OpenAI 的 tool_calls 元素。
///
/// 流式增量里字段可能只到一部分（后续 chunk 只带 `index` 和参数片段），
/// 所以除 `id` 外全是 Option；`merge_from` 负责把这些碎片拼成完整项。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    /// 流式增量中的序号，用于把同一条调用的碎片归并到一起。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// OpenAI 嵌套格式：`{"name": ..., "arguments": "<JSON 文本>"}`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<WireToolCallFunction>,
    /// 平铺格式，部分网关直接给 `name` / `arguments`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireToolCallFunction {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

impl WireToolCall {
    /// 把一条增量并进本条：同一条调用的字段可能分散在多个 chunk 里。
    pub fn merge_from(&mut self, delta: &WireToolCall) {
        if !delta.id.is_empty() {
            self.id = delta.id.clone();
        }
        if delta.call_type.is_some() {
            self.call_type = delta.call_type.clone();
        }
        if delta.index.is_some() {
            self.index = delta.index;
        }
        if let Some(delta_function) = &delta.function {
            match &mut self.function {
                Some(existing) => {
                    if !delta_function.name.is_empty() {
                        existing.name = delta_function.name.clone();
                    }
                    existing.arguments.push_str(&delta_function.arguments);
                }
                None => self.function = Some(delta_function.clone()),
            }
        }
        if let Some(delta_name) = &delta.name
            && !delta_name.is_empty()
        {
            self.name = Some(delta_name.clone());
        }
        if delta.arguments.is_some() {
            self.arguments = delta.arguments.clone();
        }
    }

    /// 两条增量是否指向同一条调用：优先按 `index`，否则按非空 `id`。
    pub fn same_call(&self, other: &WireToolCall) -> bool {
        match (self.index, other.index) {
            (Some(left), Some(right)) => left == right,
            _ => !self.id.is_empty() && self.id == other.id,
        }
    }

    /// wire → 域。
    ///
    /// 参数在 wire 上是 JSON 文本，解析失败时记录告警并落 [`Value::Null`]。
    pub fn into_domain(self) -> ToolCall {
        let (name, raw_arguments) = match &self.function {
            Some(function) => (function.name.clone(), Some(function.arguments.as_str())),
            None => (self.name.clone().unwrap_or_default(), None),
        };

        let arguments = match (raw_arguments, &self.arguments) {
            (Some(raw), _) if !raw.trim().is_empty() => match serde_json::from_str(raw) {
                Ok(parsed) => parsed,
                Err(error) => {
                    warn!(%error, raw, "tool call arguments are not valid JSON");
                    Value::Null
                }
            },
            (_, Some(parsed)) => parsed.clone(),
            _ => Value::Null,
        };

        ToolCall {
            id: self.id,
            name,
            arguments,
        }
    }
}

impl From<&ToolCall> for WireToolCall {
    fn from(call: &ToolCall) -> Self {
        Self {
            id: call.id.clone(),
            call_type: Some("function".to_string()),
            index: None,
            function: Some(WireToolCallFunction {
                name: call.name.clone(),
                arguments: call.arguments.to_string(),
            }),
            name: None,
            arguments: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WireTool {
    #[serde(rename = "type")]
    pub tool_type: &'static str,
    pub function: WireToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum WireToolChoice {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        choice_type: &'static str,
        function: WireToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct WireToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WireResponseFormat {
    JsonObject,
    JsonSchema { json_schema: WireJsonSchema },
}

#[derive(Debug, Clone, Serialize)]
pub struct WireJsonSchema {
    pub name: String,
    pub schema: Value,
    pub strict: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireThinking {
    #[serde(rename = "type")]
    pub thinking_type: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireStreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct WireAudioConfig {
    pub format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
}

/// 域请求 → wire body。所有厂商私有字段的编码都发生在这里。
impl TryFrom<&GenerateRequest> for WireChatRequest {
    type Error = ModelError;

    fn try_from(request: &GenerateRequest) -> Result<Self, Self::Error> {
        let messages = wire_messages_from_domain(&request.messages)?;

        let tools = request.tools.as_ref().map(|defs| {
            defs.iter()
                .map(|def| WireTool {
                    tool_type: "function",
                    function: WireToolFunction {
                        name: def.name.clone(),
                        description: def.description.clone(),
                        parameters: def.parameters.clone(),
                    },
                })
                .collect()
        });

        let tool_choice = request.tool_choice.as_ref().map(|choice| match choice {
            ToolChoice::Auto => WireToolChoice::Mode("auto"),
            ToolChoice::None => WireToolChoice::Mode("none"),
            ToolChoice::Required => WireToolChoice::Mode("required"),
            ToolChoice::Tool { name } => WireToolChoice::Function {
                choice_type: "function",
                function: WireToolChoiceFunction { name: name.clone() },
            },
        });

        let response_format = match &request.response_format {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(WireResponseFormat::JsonObject),
            ResponseFormat::JsonSchema {
                name,
                schema,
                strict,
            } => Some(WireResponseFormat::JsonSchema {
                json_schema: WireJsonSchema {
                    name: name.clone(),
                    schema: schema.clone(),
                    strict: *strict,
                },
            }),
        };

        let (reasoning_effort, thinking) = match request.reasoning {
            ReasoningConfig::Default => (None, None),
            ReasoningConfig::Disabled => (
                None,
                Some(WireThinking {
                    thinking_type: "disabled",
                }),
            ),
            ReasoningConfig::Enabled | ReasoningConfig::Budget { .. } => (
                None,
                Some(WireThinking {
                    thinking_type: "enabled",
                }),
            ),
            ReasoningConfig::Effort { level } => (Some(level.as_str().to_string()), None),
        };

        let stream_options = request
            .include_usage
            .then_some(WireStreamOptions { include_usage: true });

        let is_plain_text = request.output_modalities.is_empty()
            || request.output_modalities == [Modality::Text];

        let mut modalities = None;
        let mut audio = None;
        if !is_plain_text {
            let mut wire_modalities = Vec::new();
            let wants_audio = request.output_modalities.contains(&Modality::Audio);
            if wants_audio && !request.output_modalities.contains(&Modality::Text) {
                wire_modalities.push("text".to_string());
            }
            for modality in &request.output_modalities {
                let name = match modality {
                    Modality::Text => "text",
                    Modality::Image => "image",
                    Modality::Audio => "audio",
                    other => {
                        return Err(ModelError::Unsupported(format!(
                            "OpenAI chat completions does not support `{other}` output modality"
                        )));
                    }
                };
                if !wire_modalities.iter().any(|existing| existing == name) {
                    wire_modalities.push(name.to_string());
                }
            }
            modalities = Some(wire_modalities);

            if wants_audio {
                let format = request
                    .audio_options
                    .as_ref()
                    .and_then(|o| o.format.clone())
                    .unwrap_or_else(|| "wav".to_string());
                let voice = request.audio_options.as_ref().and_then(|o| o.voice.clone());
                audio = Some(WireAudioConfig { format, voice });
            }
        }

        let extra = match &request.provider_options {
            None => None,
            Some(Value::Object(map)) => Some(map.clone()),
            Some(_) => {
                return Err(ModelError::InvalidRequest(
                    "provider_options must be a JSON object".to_string(),
                ));
            }
        };

        Ok(Self {
            model: request.model.clone(),
            messages,
            stream: None,
            temperature: request.temperature,
            top_p: request.top_p,
            max_tokens: request.max_tokens,
            stop: request.stop_sequences.clone(),
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            thinking,
            stream_options,
            modalities,
            audio,
            extra,
        })
    }
}

// ============================================================================
// Token 使用统计
// ============================================================================

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct WirePromptTokensDetails {
    #[serde(default, deserialize_with = "deserialize_option_u32_lossy")]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct WireCompletionTokensDetails {
    #[serde(default, deserialize_with = "deserialize_option_u32_lossy")]
    pub reasoning_tokens: Option<u32>,
}

/// Token 使用统计：兼容 OpenAI、DeepSeek 与 OpenRouter 字段变体。
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireUsage {
    #[serde(alias = "input_tokens")]
    #[serde(deserialize_with = "deserialize_u32_lossy")]
    pub prompt_tokens: u32,
    #[serde(alias = "output_tokens")]
    #[serde(deserialize_with = "deserialize_u32_lossy")]
    pub completion_tokens: u32,
    #[serde(
        default,
        deserialize_with = "deserialize_option_u32_lossy",
        skip_serializing_if = "Option::is_none"
    )]
    pub total_tokens: Option<u32>,
    #[serde(
        default,
        deserialize_with = "deserialize_option_u32_lossy",
        skip_serializing_if = "Option::is_none"
    )]
    pub prompt_cache_hit_tokens: Option<u32>,
    #[serde(
        default,
        deserialize_with = "deserialize_option_u32_lossy",
        skip_serializing_if = "Option::is_none"
    )]
    pub prompt_cache_miss_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<WirePromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<WireCompletionTokensDetails>,
}

impl WireUsage {
    pub fn total(&self) -> u32 {
        self.total_tokens
            .unwrap_or(self.prompt_tokens + self.completion_tokens)
    }
}

impl From<WireUsage> for crate::Usage {
    fn from(value: WireUsage) -> Self {
        let total = value.total();
        let cache_read_tokens = value.prompt_cache_hit_tokens.or_else(|| {
            value
                .prompt_tokens_details
                .as_ref()
                .and_then(|d| d.cached_tokens)
        });
        let reasoning_tokens = value
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens);

        Self {
            input_tokens: value.prompt_tokens,
            output_tokens: value.completion_tokens,
            total_tokens: total,
            cache_read_tokens,
            cache_write_tokens: None,
            reasoning_tokens,
        }
    }
}

// ============================================================================
// 响应
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireChoiceImgUrl {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireChoiceImg {
    #[serde(rename = "type")]
    pub img_type: String,
    pub image_url: WireChoiceImgUrl,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireChoiceAudio {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// 选择项中的消息（非流式响应）
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireChoiceMessage {
    pub role: MessageRole,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// DeepSeek 和 OpenRouter 的推理内容。
    #[serde(alias = "reasoning", skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<WireChoiceImg>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<WireChoiceAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// 非流式响应的选择项
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireChoice {
    #[serde(default, deserialize_with = "deserialize_u32_lossy")]
    pub index: u32,
    pub message: WireChoiceMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// 非流式完整响应
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub model: String,
    pub choices: Vec<WireChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<WireUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

// ============================================================================
// 流式响应
// ============================================================================

/// 流式响应中的 delta 内容
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct WireDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<MessageRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// DeepSeek 和 OpenRouter 的推理内容。
    #[serde(alias = "reasoning", skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<WireChoiceImg>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<WireChoiceAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,
}

/// 流式响应的选择项
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireStreamChoice {
    #[serde(default, deserialize_with = "deserialize_u32_lossy")]
    pub index: u32,
    #[serde(default)]
    pub delta: WireDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

/// 流式响应块（元数据字段宽松默认，兼容各网关的尾部 usage-only 帧）
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireStreamResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    #[serde(default)]
    pub choices: Vec<WireStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<WireUsage>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;
    use crate::capability::ReasoningEffort;
    use serde_json::json;

    #[test]
    fn tool_call_fragments_merge_by_index() {
        let mut accumulated = WireToolCall {
            id: "call_1".to_string(),
            call_type: Some("function".to_string()),
            index: Some(0),
            function: Some(WireToolCallFunction {
                name: "read_file".to_string(),
                arguments: String::new(),
            }),
            ..Default::default()
        };

        let fragment_a = WireToolCall {
            index: Some(0),
            function: Some(WireToolCallFunction {
                name: String::new(),
                arguments: "{\"path\":".to_string(),
            }),
            ..Default::default()
        };
        let fragment_b = WireToolCall {
            index: Some(0),
            function: Some(WireToolCallFunction {
                name: String::new(),
                arguments: "\"a.txt\"}".to_string(),
            }),
            ..Default::default()
        };

        assert!(accumulated.same_call(&fragment_a));
        accumulated.merge_from(&fragment_a);
        accumulated.merge_from(&fragment_b);

        assert_eq!(
            accumulated.function.unwrap().arguments,
            "{\"path\":\"a.txt\"}"
        );
    }

    #[test]
    fn wire_tool_call_maps_to_domain_with_parsed_arguments() {
        let call = WireToolCall {
            id: "call_1".to_string(),
            call_type: Some("function".to_string()),
            index: Some(0),
            function: Some(WireToolCallFunction {
                name: "read_file".to_string(),
                arguments: "{\"path\":\"a.txt\"}".to_string(),
            }),
            name: None,
            arguments: None,
        };

        assert_eq!(
            call.into_domain(),
            ToolCall {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                arguments: json!({ "path": "a.txt" }),
            }
        );
    }

    #[test]
    fn malformed_tool_arguments_fall_back_to_null() {
        let call = WireToolCall {
            id: "call_1".to_string(),
            function: Some(WireToolCallFunction {
                name: "read_file".to_string(),
                arguments: "{\"path\":".to_string(),
            }),
            ..Default::default()
        };

        assert_eq!(call.into_domain().arguments, Value::Null);
    }

    #[test]
    fn domain_tool_call_encodes_back_to_nested_wire_shape() {
        let call = ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({ "path": "a.txt" }),
        };

        let wire = serde_json::to_value(WireToolCall::from(&call)).unwrap();

        assert_eq!(wire["id"], "call_1");
        assert_eq!(wire["type"], "function");
        assert_eq!(wire["function"]["name"], "read_file");
        assert_eq!(wire["function"]["arguments"], "{\"path\":\"a.txt\"}");
        assert!(wire.get("index").is_none());
    }

    #[test]
    fn reasoning_accepts_openrouter_and_deepseek_responses() {
        for field in ["reasoning", "reasoning_content"] {
            let mut message = json!({ "role": "assistant", "content": "answer" });
            message[field] = json!("reasoning text");
            let full: WireChoiceMessage = serde_json::from_value(message.clone()).unwrap();
            let delta: WireDelta = serde_json::from_value(message).unwrap();
            assert_eq!(full.reasoning_content.as_deref(), Some("reasoning text"));
            assert_eq!(delta.reasoning_content.as_deref(), Some("reasoning text"));
        }
        let delta: WireDelta = serde_json::from_value(json!({"reasoning": null})).unwrap();
        assert!(delta.reasoning_content.is_none());
    }

    #[test]
    fn test_response_allows_null_content_for_images() {
        let body = r#"{
            "id": "resp_123",
            "object": "chat.completion",
            "created": 1743916800,
            "model": "black-forest-labs/flux.2-klein-4b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image.png"
                                }
                            }
                        ]
                    },
                    "finish_reason": "stop"
                }
            ]
        }"#;

        let response: WireResponse = serde_json::from_str(body).unwrap();
        let choice = response.choices.into_iter().next().unwrap();

        assert_eq!(choice.message.content, None);
        assert_eq!(
            choice.message.images.unwrap()[0].image_url.url,
            "https://example.com/image.png"
        );
    }

    #[test]
    fn test_response_allows_float_u32_fields() {
        let body = r#"{
            "id": "resp_123",
            "object": "chat.completion",
            "created": 1743916800,
            "model": "black-forest-labs/flux.2-klein-4b",
            "choices": [
                {
                    "index": 14417.92,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "images": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://example.com/image.png"
                                }
                            }
                        ]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 1.2,
                "completion_tokens": "2.7",
                "total_tokens": 3.9,
                "prompt_cache_hit_tokens": "4",
                "prompt_cache_miss_tokens": null
            }
        }"#;

        let response: WireResponse = serde_json::from_str(body).unwrap();
        let choice = response.choices.into_iter().next().unwrap();
        let usage = response.usage.unwrap();

        assert_eq!(choice.index, 14418);
        assert_eq!(usage.prompt_tokens, 1);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.total_tokens, Some(4));
        assert_eq!(usage.prompt_cache_hit_tokens, Some(4));
        assert_eq!(usage.prompt_cache_miss_tokens, None);
    }

    #[test]
    fn test_stream_response_allows_audio_delta() {
        let body = r#"{
            "id": "chunk_123",
            "object": "chat.completion.chunk",
            "created": 1743916800,
            "model": "google/lyria-3-clip-preview",
            "system_fingerprint": null,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "audio": {
                            "data": "YWJj",
                            "transcript": "hello"
                        }
                    },
                    "finish_reason": null
                }
            ]
        }"#;

        let response: WireStreamResponse = serde_json::from_str(body).unwrap();
        let audio = response.choices[0].delta.audio.as_ref().unwrap();

        assert_eq!(audio.data.as_deref(), Some("YWJj"));
        assert_eq!(audio.transcript.as_deref(), Some("hello"));
    }

    #[test]
    fn generate_request_maps_domain_fields_to_wire_body() {
        let request = GenerateRequest::new("deepseek-v4-pro", vec![Message::user("hello")])
            .with_temperature(0.25)
            .with_max_tokens(128)
            .with_response_format(ResponseFormat::JsonObject)
            .with_reasoning(ReasoningConfig::Effort {
                level: ReasoningEffort::High,
            })
            .with_stream_usage(true)
            .with_tools(Some(vec![crate::message::ToolDef {
                name: "read_file".to_string(),
                description: "read".to_string(),
                parameters: json!({ "type": "object" }),
            }]));

        let wire = serde_json::to_value(WireChatRequest::try_from(&request).unwrap()).unwrap();

        assert_eq!(wire["model"], "deepseek-v4-pro");
        assert_eq!(wire["temperature"], 0.25);
        assert_eq!(wire["max_tokens"], 128);
        assert_eq!(wire["response_format"]["type"], "json_object");
        assert_eq!(wire["reasoning_effort"], "high");
        assert_eq!(wire["stream_options"]["include_usage"], true);
        assert_eq!(wire["tools"][0]["type"], "function");
        assert_eq!(wire["tools"][0]["function"]["name"], "read_file");
        assert!(wire.get("stream").is_none());
        assert!(wire.get("modalities").is_none());
    }
}
