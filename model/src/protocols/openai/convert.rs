//! 域模型与 OpenAI Chat Completions 协议类型之间的双向映射。

use serde_json::Value;
use tracing::warn;

use crate::capability::{
    GenerateRequest, ModelError, ReasoningConfig, TextFormat, ToolChoice,
};
use crate::message::{
    ContentPart, MediaSource, Message, MessageRole, Modality, ToolCall, audio_format_to_mime,
    mime_to_audio_format,
};

use super::types;

// ============================================================================
// 域请求 → ChatCompletionRequest
// ============================================================================

fn message_content_from_parts(parts: &[ContentPart]) -> Result<types::MessageContent, ModelError> {
    if parts.is_empty() {
        return Ok(types::MessageContent::Text(String::new()));
    }

    if parts
        .iter()
        .all(|part| matches!(part, ContentPart::Text { .. }))
    {
        if let [ContentPart::Text { text }] = parts {
            return Ok(types::MessageContent::Text(text.clone()));
        }
        let joined = parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Ok(types::MessageContent::Text(joined));
    }

    let mut protocol_parts = Vec::with_capacity(parts.len());
    for part in parts {
        match part {
            ContentPart::Text { text } => {
                protocol_parts.push(types::ContentPart::Text { text: text.clone() });
            }
            ContentPart::Image { source, detail } => {
                protocol_parts.push(types::ContentPart::ImageUrl {
                    image_url: types::ImageUrl {
                        url: source.to_uri().into_owned(),
                        detail: detail.clone(),
                    },
                });
            }
            ContentPart::Audio { source, .. } => match source {
                MediaSource::Base64 { media_type, data } => {
                    protocol_parts.push(types::ContentPart::InputAudio {
                        input_audio: types::InputAudio {
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
                    MediaSource::Base64 { .. } => types::InputFile {
                        filename: filename.clone(),
                        file_data: Some(source.to_uri().into_owned()),
                        file_id: None,
                    },
                    MediaSource::FileRef { uri, .. } => types::InputFile {
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
                protocol_parts.push(types::ContentPart::File { file });
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

    Ok(types::MessageContent::Parts(protocol_parts))
}

fn chat_messages_from_domain(messages: &[Message]) -> Result<Vec<types::ChatMessage>, ModelError> {
    let mut out = Vec::with_capacity(messages.len());

    for message in messages {
        match message.role {
            MessageRole::System => {
                out.push(types::ChatMessage::System {
                    content: message.text().into_owned(),
                });
            }
            MessageRole::User => {
                out.push(types::ChatMessage::User {
                    content: message_content_from_parts(&message.content)?,
                });
            }
            MessageRole::Assistant => {
                let content = message.text().into_owned();
                let reasoning_content = message.reasoning();
                let tool_calls: Vec<types::ToolCall> = message
                    .tool_calls()
                    .into_iter()
                    .map(types::ToolCall::from)
                    .collect();
                out.push(types::ChatMessage::Assistant {
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
                        out.push(types::ChatMessage::Tool {
                            tool_call_id: tool_call_id.clone(),
                            content: message_content_from_parts(content)?,
                        });
                        emitted = true;
                    }
                }
                if !emitted {
                    out.push(types::ChatMessage::Tool {
                        tool_call_id: String::new(),
                        content: message_content_from_parts(&message.content)?,
                    });
                }
            }
        }
    }

    Ok(out)
}

/// 域请求 → OpenAI `ChatCompletionRequest`。
impl TryFrom<&GenerateRequest> for types::ChatCompletionRequest {
    type Error = ModelError;

    fn try_from(request: &GenerateRequest) -> Result<Self, Self::Error> {
        let config = &request.config;
        let messages = chat_messages_from_domain(&request.messages)?;

        let tools = request.tools.as_ref().map(|defs| {
            defs.iter()
                .map(|def| types::Tool {
                    tool_type: "function",
                    function: types::ToolFunction {
                        name: def.name.clone(),
                        description: def.description.clone(),
                        parameters: def.parameters.clone(),
                    },
                })
                .collect()
        });

        let tool_choice = config.tool_choice.as_ref().map(|choice| match choice {
            ToolChoice::Auto => types::ToolChoice::Mode("auto"),
            ToolChoice::None => types::ToolChoice::Mode("none"),
            ToolChoice::Required => types::ToolChoice::Mode("required"),
            ToolChoice::Tool { name } => types::ToolChoice::Function {
                choice_type: "function",
                function: types::ToolChoiceFunction { name: name.clone() },
            },
        });

        let response_format = match config.text_format() {
            None | Some(TextFormat::Plain) => None,
            Some(TextFormat::JsonObject) => Some(types::ResponseFormat::JsonObject),
            Some(TextFormat::JsonSchema {
                name,
                schema,
                strict,
            }) => Some(types::ResponseFormat::JsonSchema {
                json_schema: types::JsonSchema {
                    name: name.clone(),
                    schema: schema.clone(),
                    strict: *strict,
                },
            }),
        };

        let (reasoning_effort, thinking) = match config.reasoning {
            ReasoningConfig::Default => (None, None),
            ReasoningConfig::Disabled => (
                None,
                Some(types::Thinking {
                    thinking_type: "disabled",
                }),
            ),
            ReasoningConfig::Enabled | ReasoningConfig::Budget { .. } => (
                None,
                Some(types::Thinking {
                    thinking_type: "enabled",
                }),
            ),
            ReasoningConfig::Effort { level } => (Some(level.as_str().to_string()), None),
        };

        let output_modalities = request.output_modalities();
        let is_plain_text = output_modalities == [Modality::Text];

        let mut modalities = None;
        let mut audio = None;
        if !is_plain_text {
            let mut target_modalities = Vec::new();
            let wants_audio = output_modalities.contains(&Modality::Audio);
            if wants_audio && !output_modalities.contains(&Modality::Text) {
                target_modalities.push("text".to_string());
            }
            for modality in &output_modalities {
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
                if !target_modalities.iter().any(|existing| existing == name) {
                    target_modalities.push(name.to_string());
                }
            }
            modalities = Some(target_modalities);

            if wants_audio {
                let format = config
                    .audio_options()
                    .and_then(|o| o.format.clone())
                    .unwrap_or_else(|| "wav".to_string());
                let voice = config.audio_options().and_then(|o| o.voice.clone());
                audio = Some(types::AudioConfig { format, voice });
            }
        }

        let extra = match &config.provider_options {
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
            temperature: config.temperature,
            top_p: config.top_p,
            max_tokens: config.max_tokens,
            stop: config.stop_sequences.clone(),
            tools,
            tool_choice,
            response_format,
            reasoning_effort,
            thinking,
            stream_options: None,
            modalities,
            audio,
            extra,
        })
    }
}

// ============================================================================
// ToolCall / Usage / ChoiceMessage 映射
// ============================================================================

impl From<&ToolCall> for types::ToolCall {
    fn from(call: &ToolCall) -> Self {
        Self {
            id: call.id.clone(),
            call_type: Some("function".to_string()),
            index: None,
            function: Some(types::ToolCallFunction {
                name: call.name.clone(),
                arguments: call.arguments.to_string(),
            }),
            name: None,
            arguments: None,
        }
    }
}

/// OpenAI `ToolCall` → 域 [`ToolCall`]。
///
/// 参数在协议上是 JSON 文本，解析失败时记录告警并落 [`Value::Null`]。
impl From<types::ToolCall> for ToolCall {
    fn from(call: types::ToolCall) -> Self {
        let (name, raw_arguments) = match &call.function {
            Some(function) => (function.name.clone(), Some(function.arguments.as_str())),
            None => (call.name.clone().unwrap_or_default(), None),
        };

        let arguments = match (raw_arguments, &call.arguments) {
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
            id: call.id,
            name,
            arguments,
        }
    }
}

impl From<types::Usage> for crate::Usage {
    fn from(value: types::Usage) -> Self {
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

fn role_to_domain(role: &str) -> MessageRole {
    match role {
        "system" | "developer" => MessageRole::System,
        "user" => MessageRole::User,
        "tool" => MessageRole::Tool,
        _ => MessageRole::Assistant,
    }
}

pub(super) fn choice_message_to_domain(
    message: types::ChoiceMessage,
    default_audio_format: &str,
) -> Message {
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
        let format = audio.format.as_deref().unwrap_or(default_audio_format);
        parts.push(ContentPart::Audio {
            source: MediaSource::base64(audio_format_to_mime(format), data),
            transcript: audio.transcript,
        });
    }

    if let Some(tool_calls) = message.tool_calls {
        for call in tool_calls {
            parts.push(ContentPart::ToolCall(call.into()));
        }
    }

    let role = role_to_domain(&message.role);
    if role == MessageRole::Tool {
        return Message::tool_result_parts(message.tool_call_id.unwrap_or_default(), parts, false);
    }

    Message::new(role, parts)
}
