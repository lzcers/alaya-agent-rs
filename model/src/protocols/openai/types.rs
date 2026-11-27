//! OpenAI Chat Completions 协议（`POST /chat/completions`）的数据结构定义。
//!
//! 本模块只描述目标协议本身的 JSON 形状与容错反序列化规则，不依赖项目内的
//! 域模型（`capability` / `message`）。域类型与本模块之间的双向映射集中在
//! [`super::convert`]。

use serde::de::{self, Unexpected};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};

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
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
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
    pub tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<AudioConfig>,
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub extra: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputAudio {
    pub data: String,
    pub format: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputFile {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    InputAudio { input_audio: InputAudio },
    File { file: InputFile },
}

/// 纯文本直接序列化为字符串，多模态序列化为 OpenAI content parts 数组。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// 请求体里的消息。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum ChatMessage {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: MessageContent },
    #[serde(rename = "assistant")]
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: MessageContent,
    },
}

/// OpenAI 的 `tool_calls` 元素。
///
/// 流式增量里字段可能只到一部分（后续 chunk 只带 `index` 和参数片段），
/// 所以除 `id` 外全是 Option；`merge_from` 负责把这些碎片拼成完整项。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    /// 流式增量中的序号，用于把同一条调用的碎片归并到一起。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// OpenAI 嵌套格式：`{"name": ..., "arguments": "<JSON 文本>"}`
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunction>,
    /// 平铺格式，部分网关直接给 `name` / `arguments`。
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallFunction {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

impl ToolCall {
    /// 把一条增量并进本条：同一条调用的字段可能分散在多个 chunk 里。
    pub fn merge_from(&mut self, delta: &ToolCall) {
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
    pub fn same_call(&self, other: &ToolCall) -> bool {
        match (self.index, other.index) {
            (Some(left), Some(right)) => left == right,
            _ => !self.id.is_empty() && self.id == other.id,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tool_type: &'static str,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(&'static str),
    Function {
        #[serde(rename = "type")]
        choice_type: &'static str,
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    JsonObject,
    JsonSchema { json_schema: JsonSchema },
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonSchema {
    pub name: String,
    pub schema: Value,
    pub strict: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Thinking {
    #[serde(rename = "type")]
    pub thinking_type: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioConfig {
    pub format: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
}

// ============================================================================
// Token 使用统计
// ============================================================================

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PromptTokensDetails {
    #[serde(default, deserialize_with = "deserialize_option_u32_lossy")]
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CompletionTokensDetails {
    #[serde(default, deserialize_with = "deserialize_option_u32_lossy")]
    pub reasoning_tokens: Option<u32>,
}

/// Token 使用统计：兼容 OpenAI、DeepSeek 与 OpenRouter 字段变体。
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Usage {
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
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

impl Usage {
    pub fn total(&self) -> u32 {
        self.total_tokens
            .unwrap_or(self.prompt_tokens + self.completion_tokens)
    }
}

// ============================================================================
// 非流式响应
// ============================================================================

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChoiceImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChoiceImage {
    #[serde(rename = "type")]
    pub img_type: String,
    pub image_url: ChoiceImageUrl,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChoiceAudio {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// 选择项中的消息（非流式响应）。
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChoiceMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// DeepSeek 和 OpenRouter 的推理内容。
    #[serde(alias = "reasoning", skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ChoiceImage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChoiceAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// 非流式响应的选择项。
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatChoice {
    #[serde(default, deserialize_with = "deserialize_u32_lossy")]
    pub index: u32,
    pub message: ChoiceMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// 非流式完整响应。
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    #[serde(default)]
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_fingerprint: Option<String>,
}

// ============================================================================
// 流式响应
// ============================================================================

/// 流式响应中的 delta 内容。
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// DeepSeek 和 OpenRouter 的推理内容。
    #[serde(alias = "reasoning", skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<ChoiceImage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio: Option<ChoiceAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// 流式响应的选择项。
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChunkChoice {
    #[serde(default, deserialize_with = "deserialize_u32_lossy")]
    pub index: u32,
    #[serde(default)]
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

/// 流式响应块（元数据字段宽松默认，兼容各网关的尾部 usage-only 帧）。
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionChunk {
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
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}
