//! 统一的多模态消息域（IR）。
//!
//! 这是与厂商 wire 形状完全解耦的纯域类型：所有角色（System / User / Assistant / Tool）
//! 共享同一套 [`ContentPart`] 模型，统一表达输入与输出的文本、图片、音频、视频、文件、
//! 推理过程（含签名透传）、工具调用与多模态工具结果。

pub mod tool;

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

pub use tool::{ToolCall, ToolDef};

/// 输入 / 输出模态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Image,
    Audio,
    Video,
    File,
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::File => "file",
        };
        f.write_str(label)
    }
}

/// 消息角色。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

/// 媒体数据载体。
///
/// 显式区分远程 URL、内联 base64（带 MIME 类型）和厂商文件 API 引用，
/// 避免把所有二进制模态硬编码成 `data:` URL 字符串。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MediaSource {
    /// 远程地址（`https://...`）
    Url { url: String },
    /// 内联 base64 编码数据与显式 MIME 类型（如 `image/png`、`audio/wav`、`application/pdf`）
    Base64 { media_type: String, data: String },
    /// 厂商文件 API 引用（如 Gemini `fileUri`、OpenAI file id）或本地资产引用
    FileRef { uri: String, media_type: String },
}

impl MediaSource {
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url { url: url.into() }
    }

    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        Self::Base64 {
            media_type: media_type.into(),
            data: data.into(),
        }
    }

    pub fn file_ref(uri: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::FileRef {
            uri: uri.into(),
            media_type: media_type.into(),
        }
    }

    /// 从 URL 或 `data:<mime>;base64,<payload>` 构造 [`MediaSource`]。
    pub fn from_uri(uri: impl Into<String>) -> Self {
        let uri = uri.into();
        if let Some(rest) = uri.strip_prefix("data:")
            && let Some((meta, data)) = rest.split_once(',')
            && let Some(media_type) = meta.strip_suffix(";base64")
        {
            return Self::Base64 {
                media_type: if media_type.is_empty() {
                    "application/octet-stream".to_string()
                } else {
                    media_type.to_string()
                },
                data: data.to_string(),
            };
        }
        Self::Url { url: uri }
    }

    /// 转换为可展示或供 OpenAI 风格接口使用的 URI（`https://...` 或 `data:<mime>;base64,...`）。
    pub fn to_uri(&self) -> Cow<'_, str> {
        match self {
            Self::Url { url } => Cow::Borrowed(url.as_str()),
            Self::Base64 { media_type, data } => {
                Cow::Owned(format!("data:{media_type};base64,{data}"))
            }
            Self::FileRef { uri, .. } => Cow::Borrowed(uri.as_str()),
        }
    }

    pub fn media_type(&self) -> Option<&str> {
        match self {
            Self::Url { .. } => None,
            Self::Base64 { media_type, .. } | Self::FileRef { media_type, .. } => {
                Some(media_type.as_str())
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Url { url } => url.trim().is_empty(),
            Self::Base64 { data, .. } => data.trim().is_empty(),
            Self::FileRef { uri, .. } => uri.trim().is_empty(),
        }
    }
}

/// 统一的内容块：覆盖所有输入输出模态与模型控制原语。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    Image {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    Audio {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        transcript: Option<String>,
    },
    Video {
        source: MediaSource,
    },
    File {
        source: MediaSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    /// 模型推理内容块；`signature` 用于 Anthropic / Gemini 等协议在多轮回传时原样透传。
    Reasoning {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        redacted: bool,
    },
    ToolCall(ToolCall),
    /// 工具执行结果，天然支持返回文本、图片、音频、文件混合内容。
    ToolResult {
        tool_call_id: String,
        content: Vec<ContentPart>,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        is_error: bool,
    },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn image(source: MediaSource) -> Self {
        Self::Image {
            source,
            detail: None,
        }
    }

    pub fn image_url(url: impl Into<String>) -> Self {
        Self::Image {
            source: MediaSource::from_uri(url),
            detail: None,
        }
    }

    pub fn image_url_with_detail(url: impl Into<String>, detail: impl Into<String>) -> Self {
        Self::Image {
            source: MediaSource::from_uri(url),
            detail: Some(detail.into()),
        }
    }

    pub fn audio(source: MediaSource) -> Self {
        Self::Audio {
            source,
            transcript: None,
        }
    }

    pub fn audio_with_transcript(source: MediaSource, transcript: impl Into<String>) -> Self {
        Self::Audio {
            source,
            transcript: Some(transcript.into()),
        }
    }

    pub fn video(source: MediaSource) -> Self {
        Self::Video { source }
    }

    pub fn file(source: MediaSource, filename: Option<String>) -> Self {
        Self::File { source, filename }
    }

    pub fn reasoning(text: impl Into<String>) -> Self {
        Self::Reasoning {
            text: text.into(),
            signature: None,
            redacted: false,
        }
    }

    pub fn tool_call(call: ToolCall) -> Self {
        Self::ToolCall(call)
    }

    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        let text = content.into();
        let parts = if text.is_empty() {
            Vec::new()
        } else {
            vec![Self::Text { text }]
        };
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            content: parts,
            is_error: false,
        }
    }

    pub fn tool_result_parts(
        tool_call_id: impl Into<String>,
        content: Vec<ContentPart>,
        is_error: bool,
    ) -> Self {
        Self::ToolResult {
            tool_call_id: tool_call_id.into(),
            content,
            is_error,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text { text } => text.trim().is_empty(),
            Self::Image { source, .. }
            | Self::Audio { source, .. }
            | Self::Video { source }
            | Self::File { source, .. } => source.is_empty(),
            Self::Reasoning {
                text,
                signature,
                redacted,
            } => text.trim().is_empty() && signature.is_none() && !redacted,
            Self::ToolCall(_) => false,
            Self::ToolResult { content, .. } => content.iter().all(Self::is_empty),
        }
    }

    /// 收集该内容块所涉及的输入模态（含 `ToolResult` 内部嵌套块）。
    pub fn collect_modalities(&self, out: &mut Vec<Modality>) {
        let push_unique = |m: Modality, out: &mut Vec<Modality>| {
            if !out.contains(&m) {
                out.push(m);
            }
        };
        match self {
            Self::Text { .. } => push_unique(Modality::Text, out),
            Self::Image { .. } => push_unique(Modality::Image, out),
            Self::Audio { .. } => push_unique(Modality::Audio, out),
            Self::Video { .. } => push_unique(Modality::Video, out),
            Self::File { .. } => push_unique(Modality::File, out),
            Self::Reasoning { .. } | Self::ToolCall(_) => {}
            Self::ToolResult { content, .. } => {
                for part in content {
                    part.collect_modalities(out);
                }
            }
        }
    }
}

/// 音频短格式名（如 `wav` / `mp3`）到 MIME 类型的映射。
pub fn audio_format_to_mime(format: &str) -> String {
    match format.to_ascii_lowercase().as_str() {
        "wav" => "audio/wav".to_string(),
        "mp3" | "mpeg" => "audio/mpeg".to_string(),
        "pcm" | "pcm16" => "audio/pcm".to_string(),
        "flac" => "audio/flac".to_string(),
        "opus" => "audio/opus".to_string(),
        "ogg" => "audio/ogg".to_string(),
        "aac" => "audio/aac".to_string(),
        other if other.starts_with("audio/") => other.to_string(),
        other => format!("audio/{other}"),
    }
}

/// MIME 类型到短音频格式名（如 `audio/mpeg` → `mp3`）的映射。
pub fn mime_to_audio_format(media_type: &str) -> &str {
    match media_type {
        "audio/wav" | "audio/x-wav" => "wav",
        "audio/mpeg" | "audio/mp3" => "mp3",
        "audio/pcm" => "pcm16",
        "audio/flac" => "flac",
        "audio/opus" => "opus",
        "audio/ogg" => "ogg",
        "audio/aac" => "aac",
        other => other.strip_prefix("audio/").unwrap_or(other),
    }
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
        render_display_text(&self.content)
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

/// 从内容块切片提取纯文本（遇到 `ToolResult` 会展开其内部文本）。
pub fn extract_text(parts: &[ContentPart]) -> Cow<'_, str> {
    let mut texts: Vec<&str> = Vec::new();
    collect_texts(parts, &mut texts);
    match texts.as_slice() {
        [] => Cow::Borrowed(""),
        [single] => Cow::Borrowed(single),
        many => Cow::Owned(many.join("\n")),
    }
}

fn collect_texts<'a>(parts: &'a [ContentPart], out: &mut Vec<&'a str>) {
    for part in parts {
        match part {
            ContentPart::Text { text } => out.push(text.as_str()),
            ContentPart::ToolResult { content, .. } => collect_texts(content, out),
            _ => {}
        }
    }
}

/// 把内容块切片渲染为可读文本视图。
pub fn render_display_text(parts: &[ContentPart]) -> String {
    let mut segments: Vec<String> = Vec::new();
    for part in parts {
        match part {
            ContentPart::Text { text } if !text.is_empty() => segments.push(text.clone()),
            ContentPart::Image { .. } => segments.push("[image]".to_string()),
            ContentPart::Audio { transcript, .. } => match transcript.as_deref() {
                Some(t) if !t.trim().is_empty() => segments.push(format!("[audio: {t}]")),
                _ => segments.push("[audio]".to_string()),
            },
            ContentPart::Video { .. } => segments.push("[video]".to_string()),
            ContentPart::File { filename, .. } => match filename.as_deref() {
                Some(name) if !name.is_empty() => segments.push(format!("[file: {name}]")),
                _ => segments.push("[file]".to_string()),
            },
            ContentPart::ToolResult { content, .. } => {
                let inner = render_display_text(content);
                if !inner.is_empty() {
                    segments.push(inner);
                }
            }
            _ => {}
        }
    }
    segments.join("\n")
}
