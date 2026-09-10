//! OpenRouter 图片协议适配器。
//!
//! OpenRouter 的 chat 与音频走 OpenAI 兼容协议（见 [`crate::protocols::openai`]），
//! 只有图片端点使用它自有的统一 schema，因此单独成模块：
//! 协议选择是 **(端点, 能力)** 的属性，不是厂商的属性。

pub mod wire;

use async_trait::async_trait;

use crate::capability::{GenImgCapability, GenImgRequest, GenImgResponse, ModelError};
use crate::providers::Provider;

use wire::{WireImageRequest, generated_image_to_url};

const IMAGES_PATH: &str = "/images";

/// OpenRouter 统一图片 API 适配器。
pub struct OpenRouterImages {
    provider: Provider,
}

impl OpenRouterImages {
    pub fn new(provider: Provider) -> Self {
        Self { provider }
    }

    pub fn provider(&self) -> &Provider {
        &self.provider
    }
}

#[async_trait]
impl GenImgCapability for OpenRouterImages {
    async fn gen_img(&self, request: GenImgRequest) -> Result<GenImgResponse, ModelError> {
        if request.prompt.trim().is_empty() {
            return Err(ModelError::NoResponse);
        }

        let wire = WireImageRequest {
            model: request.model,
            prompt: request.prompt,
            resolution: request.resolution,
            aspect_ratio: request.aspect_ratio,
        };
        let response: wire::WireImageResponse =
            self.provider.post_json(IMAGES_PATH, &wire).await?;
        let image_urls = response
            .data
            .into_iter()
            .filter_map(generated_image_to_url)
            .collect::<Vec<_>>();

        if image_urls.is_empty() {
            return Err(ModelError::NoResponse);
        }

        Ok(GenImgResponse { image_urls })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::openai::testing::{FakeTransport, fake_provider};
    use serde_json::json;

    #[tokio::test]
    async fn gen_img_uses_dedicated_images_endpoint_and_returns_data_url() {
        let transport = FakeTransport::with_json(json!({
            "data": [{ "b64_json": "aW1hZ2U=", "media_type": "image/webp" }]
        }));
        let model = OpenRouterImages::new(fake_provider(transport.clone()));

        let response = model
            .gen_img(
                GenImgRequest::new("krea/krea-2-medium-turbo", "cinematic landscape")
                    .with_resolution("1K")
                    .with_aspect_ratio("16:9"),
            )
            .await
            .unwrap();

        assert_eq!(response.image_urls, vec!["data:image/webp;base64,aW1hZ2U="]);
        assert_eq!(transport.recorded_paths(), vec!["/images"]);
        let body = transport.first_body();
        assert_eq!(body["model"], "krea/krea-2-medium-turbo");
        assert_eq!(body["prompt"], "cinematic landscape");
        assert_eq!(body["resolution"], "1K");
        assert_eq!(body["aspect_ratio"], "16:9");
        assert!(body.get("messages").is_none());
        assert!(body.get("size").is_none());
    }

    #[tokio::test]
    async fn gen_img_rejects_empty_prompt_without_calling_transport() {
        let transport = FakeTransport::with_json(json!({ "data": [] }));
        let model = OpenRouterImages::new(fake_provider(transport.clone()));

        let error = model
            .gen_img(GenImgRequest::new("m", "   "))
            .await
            .expect_err("empty prompt should be rejected");

        assert!(matches!(error, ModelError::NoResponse));
        assert!(transport.recorded().is_empty());
    }
}
