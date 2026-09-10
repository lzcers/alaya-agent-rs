//! OpenRouter 统一图片 API 的 wire 结构体。
//!
//! 请求形状是 OpenRouter 自有的：`POST /api/v1/images` 用
//! `{model, prompt, resolution, aspect_ratio}`，而 OpenAI 自己的 `/images`
//! 用的是 `size` / `quality` / `response_format`。两者不可互换，所以这里
//! 的 wire 类型刻意不放在 `protocols::openai` 下。

use serde::{Deserialize, Serialize};

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

/// 把返回的图片统一成 URL（远程地址或 `data:` 内联数据）。
pub fn generated_image_to_url(image: WireGeneratedImage) -> Option<String> {
    if let Some(url) = image.url.filter(|url| !url.trim().is_empty()) {
        return Some(url);
    }

    let encoded = image.b64_json.filter(|value| !value.trim().is_empty())?;
    let media_type = image
        .media_type
        .filter(|value| value.starts_with("image/"))
        .unwrap_or_else(|| "image/png".to_string());
    Some(format!("data:{media_type};base64,{encoded}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_image_prefers_url_and_falls_back_to_data_url() {
        assert_eq!(
            generated_image_to_url(WireGeneratedImage {
                url: Some("https://example.com/a.png".to_string()),
                b64_json: Some("ignored".to_string()),
                media_type: None,
            })
            .as_deref(),
            Some("https://example.com/a.png")
        );
        assert_eq!(
            generated_image_to_url(WireGeneratedImage {
                url: None,
                b64_json: Some("aW1hZ2U=".to_string()),
                media_type: Some("image/webp".to_string()),
            })
            .as_deref(),
            Some("data:image/webp;base64,aW1hZ2U=")
        );
        assert_eq!(
            generated_image_to_url(WireGeneratedImage {
                url: None,
                b64_json: None,
                media_type: None,
            }),
            None
        );
    }
}
