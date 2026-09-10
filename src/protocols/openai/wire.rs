//! OpenAI 兼容协议的 wire 结构体。
//!
//! 这里是「线上长什么样」的唯一真相：字段名、别名、容错解析、以及
//! 域请求 → wire body 的翻译。域层不应该看见本模块的任何类型。

use serde::de::{self, Unexpected};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use tracing::warn;

use crate::MessageRole;
use crate::capability::ChatRequest;
use crate::message::{Message, ToolCall};

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
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<WireTool>>,
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
}

/// 请求体里的消息。
///
/// 与域 [`Message`] 的唯一差别是 `tool_calls` 用 wire 形状——域侧的
/// [`ToolCall`] 只有 `{id, name, arguments}`，需要在边界上翻译。
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum WireMessage {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: String },
    #[serde(rename = "assistant")]
    Assistant {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<WireToolCall>>,
    },
    #[serde(rename = "tool")]
    Tool { tool_call_id: String, content: String },
}

impl From<&Message> for WireMessage {
    fn from(message: &Message) -> Self {
        match message {
            Message::System { content } => Self::System {
                content: content.clone(),
            },
            Message::User { content } => Self::User {
                content: content.clone(),
            },
            Message::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => Self::Assistant {
                content: content.clone(),
                reasoning_content: reasoning_content.clone(),
                tool_calls: tool_calls
                    .as_ref()
                    .map(|calls| calls.iter().map(WireToolCall::from).collect()),
            },
            Message::Tool {
                tool_call_id,
                content,
            } => Self::Tool {
                tool_call_id: tool_call_id.clone(),
                content: content.clone(),
            },
        }
    }
}

impl From<WireMessage> for Message {
    fn from(message: WireMessage) -> Self {
        match message {
            WireMessage::System { content } => Self::System { content },
            WireMessage::User { content } => Self::User { content },
            WireMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
            } => Self::Assistant {
                content,
                reasoning_content,
                tool_calls: tool_calls.map(|calls| {
                    calls
                        .into_iter()
                        .map(WireToolCall::into_domain)
                        .collect()
                }),
            },
            WireMessage::Tool {
                tool_call_id,
                content,
            } => Self::Tool {
                tool_call_id,
                content,
            },
        }
    }
}

/// OpenAI 的 tool_calls 元素。
///
/// 流式增量里字段可能只到一部分（后续 chunk 只带 `index` 和参数片段），
/// 所以除 `id` 外全是 Option；`merge_into` 负责把这些碎片拼成完整项。
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
    /// 参数在 wire 上是 JSON 文本，解析失败时记录告警并落 [`Value::Null`]——
    /// 与静默兜底相比，至少让畸形参数可见。
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
pub struct WireResponseFormat {
    #[serde(rename = "type")]
    pub response_type: &'static str,
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

impl WireChatRequest {
    fn base(model: impl Into<String>, messages: &[Message]) -> Self {
        Self {
            model: model.into(),
            messages: messages.iter().map(WireMessage::from).collect(),
            stream: None,
            temperature: None,
            max_tokens: None,
            tools: None,
            response_format: None,
            reasoning_effort: None,
            thinking: None,
            stream_options: None,
            modalities: None,
            audio: None,
        }
    }
}

/// 域请求 → wire body。所有厂商私有字段的编码都发生在这里。
impl From<&ChatRequest> for WireChatRequest {
    fn from(request: &ChatRequest) -> Self {
        let mut wire = Self::base(request.model.clone(), &request.messages);

        wire.temperature = request.temperature;
        wire.max_tokens = request.max_tokens;

        if let Some(tools) = &request.tools {
            wire.tools = Some(
                tools
                    .iter()
                    .map(|def| WireTool {
                        tool_type: "function",
                        function: WireToolFunction {
                            name: def.name.clone(),
                            description: def.description.clone(),
                            parameters: def.parameters.clone(),
                        },
                    })
                    .collect(),
            );
        }

        if request.response_format_json {
            wire.response_format = Some(WireResponseFormat {
                response_type: "json_object",
            });
        }

        if request.include_usage {
            wire.stream_options = Some(WireStreamOptions {
                include_usage: true,
            });
        }

        wire.reasoning_effort = request.reasoning.effort.clone();
        if let Some(enabled) = request.reasoning.thinking {
            wire.thinking = Some(WireThinking {
                thinking_type: if enabled { "enabled" } else { "disabled" },
            });
        }

        wire
    }
}

// ============================================================================
// Token 使用统计
// ============================================================================

/// Token 使用统计
/// 兼容 OpenAI 和 DeepSeek 的格式（prompt_tokens/input_tokens, completion_tokens/output_tokens）
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
}

impl WireUsage {
    pub fn total(&self) -> u32 {
        self.total_tokens
            .unwrap_or(self.prompt_tokens + self.completion_tokens)
    }
}

impl From<WireUsage> for crate::Usage {
    fn from(value: WireUsage) -> Self {
        Self {
            prompt_tokens: value.prompt_tokens,
            completion_tokens: value.completion_tokens,
            total_tokens: value.total(),
            prompt_cache_hit_tokens: value.prompt_cache_hit_tokens,
            prompt_cache_miss_tokens: value.prompt_cache_miss_tokens,
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
    pub tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

/// 非流式响应的选择项
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireChoice {
    #[serde(deserialize_with = "deserialize_u32_lossy")]
    pub index: u32,
    pub message: WireChoiceMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

/// 非流式完整响应
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
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
    pub audio: Option<WireChoiceAudio>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<WireToolCall>>,
}

/// 流式响应的选择项
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireStreamChoice {
    #[serde(deserialize_with = "deserialize_u32_lossy")]
    pub index: u32,
    pub delta: WireDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

/// 流式响应块
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WireStreamResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<WireStreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<WireUsage>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;
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

        // 后续 chunk 只带 index 和参数片段。
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
    fn chat_request_maps_domain_fields_to_wire_body() {
        let request = ChatRequest::new("deepseek-v4-pro", vec![Message::user("hello")])
            .with_temperature(0.25)
            .with_max_tokens(128)
            .with_response_format_json()
            .with_reasoning_effort("high")
            .with_thinking(true)
            .with_stream_usage(true)
            .with_tools(Some(vec![crate::message::ToolDef {
                name: "read_file".to_string(),
                description: "read".to_string(),
                parameters: json!({ "type": "object" }),
            }]));

        let wire = serde_json::to_value(WireChatRequest::from(&request)).unwrap();

        assert_eq!(wire["model"], "deepseek-v4-pro");
        assert_eq!(wire["temperature"], 0.25);
        assert_eq!(wire["max_tokens"], 128);
        assert_eq!(wire["response_format"]["type"], "json_object");
        assert_eq!(wire["reasoning_effort"], "high");
        assert_eq!(wire["thinking"]["type"], "enabled");
        assert_eq!(wire["stream_options"]["include_usage"], true);
        assert_eq!(wire["tools"][0]["type"], "function");
        assert_eq!(wire["tools"][0]["function"]["name"], "read_file");
        assert!(wire.get("stream").is_none());
        assert!(wire.get("modalities").is_none());
    }

}
