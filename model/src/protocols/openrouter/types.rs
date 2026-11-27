//! OpenRouter 统一图片协议（`POST /images`）的数据结构定义。
//!
//! 请求形状是 OpenRouter 自有的：`{model, prompt, resolution, aspect_ratio}`，
//! 而 OpenAI 自己的 `/images` 用的是 `size` / `quality` / `response_format`。
//! 本模块不依赖域模型。

use serde::{Deserialize, Serialize};

/// `POST /images` 的请求体。
#[derive(Debug, Clone, Serialize)]
pub struct ImageRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
}

/// `POST /images` 的响应体。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageResponse {
    #[serde(default)]
    pub data: Vec<GeneratedImage>,
}

/// 单张生成图片条目。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GeneratedImage {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}
