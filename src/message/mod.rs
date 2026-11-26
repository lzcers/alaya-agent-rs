//! 消息域：模型看到的消息与工具词汇。
//!
//! `Message` 既是域类型也是 OpenAI 兼容的 wire messages 形状——它不携带任何
//! 协议独有字段，因此不需要在协议层再镜像一份。

pub mod tool;

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

pub use tool::{ToolCall, ToolDef};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    pub fn image_url_with_detail(url: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: Some(detail.into()),
            },
        }
    }
}

/// 用户消息内容：纯文本直接序列化为字符串，多模态序列化为 OpenAI 兼容的 content parts 数组。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UserContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl UserContent {
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(text) => text.trim().is_empty(),
            Self::Parts(parts) => parts.iter().all(|part| match part {
                ContentPart::Text { text } => text.trim().is_empty(),
                ContentPart::ImageUrl { image_url } => image_url.url.trim().is_empty(),
            }),
        }
    }

    pub fn as_text(&self) -> Cow<'_, str> {
        match self {
            Self::Text(text) => Cow::Borrowed(text.as_str()),
            Self::Parts(parts) => {
                let mut iter = parts.iter().filter_map(|part| match part {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                });
                let Some(first) = iter.next() else {
                    return Cow::Borrowed("");
                };
                let Some(second) = iter.next() else {
                    return Cow::Borrowed(first);
                };
                let mut joined = String::from(first);
                joined.push('\n');
                joined.push_str(second);
                for text in iter {
                    joined.push('\n');
                    joined.push_str(text);
                }
                Cow::Owned(joined)
            }
        }
    }

    pub fn to_display_text(&self) -> Cow<'_, str> {
        match self {
            Self::Text(text) => Cow::Borrowed(text.as_str()),
            Self::Parts(parts) => {
                let rendered: Vec<&str> = parts
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text { text } => text.as_str(),
                        ContentPart::ImageUrl { .. } => "[image]",
                    })
                    .filter(|segment| !segment.is_empty())
                    .collect();
                Cow::Owned(rendered.join("\n"))
            }
        }
    }
}

impl From<String> for UserContent {
    fn from(value: String) -> Self {
        Self::Text(value)
    }
}

impl From<&str> for UserContent {
    fn from(value: &str) -> Self {
        Self::Text(value.to_owned())
    }
}

impl From<Vec<ContentPart>> for UserContent {
    fn from(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: UserContent },
    #[serde(rename = "assistant")]
    Assistant {
        content: String,
        /// DeepSeek 推理模式的推理内容（如 deepseek-reasoner）
        #[serde(skip_serializing_if = "Option::is_none")]
        reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
    },
    #[serde(rename = "tool")]
    Tool {
        tool_call_id: String,
        content: String,
    },
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::System {
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<UserContent>) -> Self {
        Self::User {
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::Assistant {
            content: content.into(),
            reasoning_content: None,
            tool_calls: None,
        }
    }

    /// 获取消息文本内容
    pub fn content(&self) -> Cow<'_, str> {
        match self {
            Self::System { content } => Cow::Borrowed(content.as_str()),
            Self::User { content } => content.as_text(),
            Self::Assistant { content, .. } => Cow::Borrowed(content.as_str()),
            Self::Tool { content, .. } => Cow::Borrowed(content.as_str()),
        }
    }

    /// 获取推理内容（仅 Assistant 消息有）
    pub fn reasoning_content(&self) -> Option<&str> {
        match self {
            Self::Assistant {
                reasoning_content, ..
            } => reasoning_content.as_deref(),
            _ => None,
        }
    }
}
