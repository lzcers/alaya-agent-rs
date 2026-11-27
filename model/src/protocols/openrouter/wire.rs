//! OpenRouter 统一图片 API 的 wire 结构体。
//!
//! 请求形状是 OpenRouter 自有的：`POST /api/v1/images` 用
//! `{model, prompt, resolution, aspect_ratio}`，而 OpenAI 自己的 `/images`
//! 用的是 `size` / `quality` / `response_format`。

use serde::{Deserialize, Serialize};

use crate::message::MediaSource;

/// `POST /images` 的请求体。
#[derive(Debug, Clone, Serialize)]
pub struct WireImageRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireImageResponse {
    #[serde(default)]
    pub data: Vec<WireGeneratedImage>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireGeneratedImage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

/// 把返回的图片映射为结构化的 [`MediaSource`]（不再强行拼装 `data:` URL 字符串）。
pub fn generated_image_to_source(image: WireGeneratedImage) -> Option<MediaSource> {
    if let Some(url) = image.url.filter(|url| !url.trim().is_empty()) {
        return Some(MediaSource::from_uri(url));
    }

    let encoded = image.b64_json.filter(|value| !value.trim().is_empty())?;
    let media_type = image
        .media_type
        .filter(|value| value.starts_with("image/"))
        .unwrap_or_else(|| "image/png".to_string());
    Some(MediaSource::base64(media_type, encoded))
}
