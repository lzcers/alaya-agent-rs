use async_trait::async_trait;

use crate::capability::ModelError;

/// 一次图片生成的域请求。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenImgRequest {
    pub model: String,
    pub prompt: String,
    pub resolution: Option<String>,
    pub aspect_ratio: Option<String>,
}

impl GenImgRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            resolution: None,
            aspect_ratio: None,
        }
    }

    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }
}

/// 图片生成结果。
///
/// 统一成 URL（远程地址或 `data:` 内联数据），调用方不需要关心厂商是返回
/// `url` 还是 `b64_json`。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenImgResponse {
    pub image_urls: Vec<String>,
}

/// 图片生成能力。
#[async_trait]
pub trait GenImgCapability: Send + Sync {
    async fn gen_img(&self, request: GenImgRequest) -> Result<GenImgResponse, ModelError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_options_belong_to_the_request() {
        let request =
            GenImgRequest::new("model", "prompt").with_aspect_ratio("16:9").with_resolution("2K");

        assert_eq!(request.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(request.resolution.as_deref(), Some("2K"));
    }
}
