//! OpenAI 兼容协议的 wire 结构体。
//!
//! 这里是「线上长什么样」的唯一真相：字段名、别名、容错解析、以及
//! 域请求 → wire body 的翻译。域层不应该看见本模块的任何类型。

use serde::de::{self, Unexpected};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

use crate::MessageRole;
use crate::agent::ToolCall;
use crate::capability::ChatRequest;
use crate::message::Message;

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
    pub messages: Vec<Message>,
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
    fn base(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
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
        let mut wire = Self::base(request.model.clone(), request.messages.clone());

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
    pub tool_calls: Option<Vec<ToolCall>>,
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
    pub tool_calls: Option<Vec<ToolCall>>,
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
            .with_tools(Some(vec![crate::agent::ToolDef {
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
