//! 统一的多模态消息域（IR）。
//!
//! 这是与厂商 wire 形状完全解耦的纯域类型：所有角色（System / User / Assistant / Tool）
//! 共享同一套 [`ContentPart`] 模型，统一表达输入与输出的文本、图片、音频、视频、文件、
//! 推理过程（含签名透传）、工具调用与多模态工具结果。

mod content;
mod tool;

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

pub use content::{ContentPart, MediaSource, Modality};
pub(crate) use content::{audio_format_to_mime, extract_text, mime_to_audio_format};
pub use tool::{ToolCall, ToolDef};

/// 消息角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// 统一的消息模型：任意角色均可承载多模态内容块。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: Vec<ContentPart>,
}

impl Message {
    pub fn new(role: MessageRole, content: Vec<ContentPart>) -> Self {
        Self { role, content }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::with_optional_text(MessageRole::System, content.into())
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::with_optional_text(MessageRole::User, content.into())
    }

    pub fn user_parts(parts: Vec<ContentPart>) -> Self {
        Self::new(MessageRole::User, parts)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::with_optional_text(MessageRole::Assistant, content.into())
    }

    pub fn assistant_parts(parts: Vec<ContentPart>) -> Self {
        Self::new(MessageRole::Assistant, parts)
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(
            MessageRole::Tool,
            vec![ContentPart::tool_result(tool_call_id, content)],
        )
    }

    pub fn tool_result_parts(
        tool_call_id: impl Into<String>,
        content: Vec<ContentPart>,
        is_error: bool,
    ) -> Self {
        Self::new(
            MessageRole::Tool,
            vec![ContentPart::tool_result_parts(
                tool_call_id,
                content,
                is_error,
            )],
        )
    }

    pub fn with_part(mut self, part: ContentPart) -> Self {
        self.content.push(part);
        self
    }

    fn with_optional_text(role: MessageRole, text: String) -> Self {
        let content = if text.is_empty() {
            Vec::new()
        } else {
            vec![ContentPart::Text { text }]
        };
        Self { role, content }
    }

    /// 提取消息中的所有纯文本内容（对 `Tool` 消息会展开 `ToolResult` 内的文本块）。
    pub fn text(&self) -> Cow<'_, str> {
        extract_text(&self.content)
    }

    /// 渲染用于日志、展示或摘要的文本视图（非文本模态用 `[image]` 等占位）。
    pub fn display_text(&self) -> String {
        ContentPart::render_display_text(&self.content)
    }

    /// 提取推理内容文本（如果包含非空 `Reasoning` 块）。
    pub fn reasoning(&self) -> Option<String> {
        let mut segments = self.content.iter().filter_map(|part| match part {
            ContentPart::Reasoning { text, .. } if !text.is_empty() => Some(text.as_str()),
            _ => None,
        });
        let first = segments.next()?;
        let mut joined = String::from(first);
        for seg in segments {
            joined.push('\n');
            joined.push_str(seg);
        }
        Some(joined)
    }

    pub fn has_reasoning(&self) -> bool {
        self.content.iter().any(|part| {
            matches!(
                part,
                ContentPart::Reasoning { text, redacted, .. }
                    if !text.trim().is_empty() || *redacted
            )
        })
    }

    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }

    pub fn images(&self) -> Vec<&MediaSource> {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Image { source, .. } => Some(source),
                _ => None,
            })
            .collect()
    }

    pub fn audios(&self) -> Vec<(&MediaSource, Option<&str>)> {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Audio { source, transcript } => Some((source, transcript.as_deref())),
                _ => None,
            })
            .collect()
    }

    pub fn modalities(&self) -> Vec<Modality> {
        let mut out = Vec::new();
        for part in &self.content {
            part.collect_modalities(&mut out);
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.content.iter().all(ContentPart::is_empty)
    }
}
