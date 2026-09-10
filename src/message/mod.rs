//! 消息域：模型看到的消息与工具词汇。
//!
//! `Message` 既是域类型也是 OpenAI 兼容的 wire messages 形状——它不携带任何
//! 协议独有字段，因此不需要在协议层再镜像一份。

pub mod tool;

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
#[serde(tag = "role")]
pub enum Message {
    #[serde(rename = "system")]
    System { content: String },
    #[serde(rename = "user")]
    User { content: String },
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

    pub fn user(content: impl Into<String>) -> Self {
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

    /// 获取消息内容
    pub fn content(&self) -> &str {
        match self {
            Self::System { content } => content,
            Self::User { content } => content,
            Self::Assistant { content, .. } => content,
            Self::Tool { content, .. } => content,
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
