//! OpenRouter 图片协议适配器。
//!
//! OpenRouter 的对话与音频走 OpenAI 兼容协议（见 [`crate::protocols::openai`]），
//! 专用图片端点使用它自有的统一 schema，但仍对外实现统一的 [`GenerateCapability`]。

pub mod wire;

use async_trait::async_trait;

use crate::capability::{
    FinishReason, GenerateCapability, GenerateRequest, GenerateResponse, GenerateStream,
    ModelError, StreamEvent,
};
use crate::message::{ContentPart, Message, MessageRole, Modality};
use crate::providers::Provider;

use wire::{WireImageRequest, generated_image_to_source};

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

fn extract_prompt(request: &GenerateRequest) -> Option<String> {
    request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::User)
        .map(|m| m.text().into_owned())
        .filter(|text| !text.trim().is_empty())
}

#[async_trait]
impl GenerateCapability for OpenRouterImages {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError> {
        if !request.output_modalities.contains(&Modality::Image) {
            return Err(ModelError::Unsupported(
                "OpenRouterImages only supports `image` output modality".to_string(),
            ));
        }

        let prompt = extract_prompt(&request)
            .ok_or_else(|| ModelError::InvalidRequest("image prompt must not be empty".into()))?;

        let (resolution, aspect_ratio) = match request.image_options {
            Some(opts) => (opts.resolution, opts.aspect_ratio),
            None => (None, None),
        };

        let model = request.model.clone();
        let wire = WireImageRequest {
            model: request.model,
            prompt,
            resolution,
            aspect_ratio,
        };
        let response: wire::WireImageResponse = self.provider.post_json(IMAGES_PATH, &wire).await?;
        let parts: Vec<ContentPart> = response
            .data
            .into_iter()
            .filter_map(generated_image_to_source)
            .map(ContentPart::image)
            .collect();

        if parts.is_empty() {
            return Err(ModelError::NoResponse);
        }

        Ok(GenerateResponse {
            message: Message::assistant_parts(parts),
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            model: Some(model),
            id: None,
        })
    }

    async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateStream, ModelError> {
        let response = self.generate(request).await?;
        let mut events: Vec<Result<StreamEvent, ModelError>> = response
            .message
            .content
            .into_iter()
            .map(|part| Ok(StreamEvent::Part(part)))
            .collect();
        if let Some(usage) = response.usage {
            events.push(Ok(StreamEvent::Usage(usage)));
        }
        events.push(Ok(StreamEvent::Finished {
            reason: response.finish_reason,
        }));
        Ok(Box::pin(futures::stream::iter(events)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::ImageOptions;
    use crate::message::MediaSource;
    use crate::protocols::openai::testing::{FakeTransport, fake_provider};
    use serde_json::json;

    #[tokio::test]
    async fn generate_uses_dedicated_images_endpoint_and_returns_media_source() {
        let transport = FakeTransport::with_json(json!({
            "data": [{ "b64_json": "aW1hZ2U=", "media_type": "image/webp" }]
        }));
        let model = OpenRouterImages::new(fake_provider(transport.clone()));

        let response = model
            .generate(
                GenerateRequest::image("krea/krea-2-medium-turbo", "cinematic landscape")
                    .with_image_options(
                        ImageOptions::new()
                            .with_resolution("1K")
                            .with_aspect_ratio("16:9"),
                    ),
            )
            .await
            .unwrap();

        assert_eq!(
            response.message.images(),
            vec![&MediaSource::base64("image/webp", "aW1hZ2U=")]
        );
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
    async fn generate_rejects_empty_prompt_without_calling_transport() {
        let transport = FakeTransport::with_json(json!({ "data": [] }));
        let model = OpenRouterImages::new(fake_provider(transport.clone()));

        let error = model
            .generate(GenerateRequest::image("m", "   "))
            .await
            .expect_err("empty prompt should be rejected");

        assert!(matches!(error, ModelError::InvalidRequest(_)));
        assert!(transport.recorded().is_empty());
    }
}
