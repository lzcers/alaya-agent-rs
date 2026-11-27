//! OpenRouter 图片协议适配器。
//!
//! OpenRouter 的对话与音频走 OpenAI 兼容协议（见 [`crate::protocols::openai`]），
//! 专用图片端点使用自有数据结构（见 [`types`]），并对外实现统一的
//! [`GenerateCapability`]。

pub mod types;

use async_trait::async_trait;

use crate::capability::{
    FinishReason, GenerateCapability, GenerateRequest, GenerateResponse, GenerateStream,
    ModelError, StreamEvent,
};
use crate::message::{ContentPart, MediaSource, Message, MessageRole};
use crate::providers::Provider;

use types::{GeneratedImage, ImageRequest, ImageResponse};

const IMAGES_PATH: &str = "/images";

/// OpenRouter 统一图片 API 适配器。
pub struct OpenRouterImages {
    provider: Provider,
}

impl OpenRouterImages {
    pub fn new(provider: Provider) -> Self {
        Self { provider }
    }

    #[cfg(test)]
    pub fn provider(&self) -> &Provider {
        &self.provider
    }

    /// `POST /images`。
    pub async fn create_image(&self, request: &ImageRequest) -> Result<ImageResponse, ModelError> {
        Ok(self.provider.post_json(IMAGES_PATH, request).await?)
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

impl TryFrom<&GenerateRequest> for ImageRequest {
    type Error = ModelError;

    fn try_from(request: &GenerateRequest) -> Result<Self, Self::Error> {
        let Some(opts) = request.config.image_options() else {
            return Err(ModelError::Unsupported(
                "OpenRouterImages only supports `image` output modality".to_string(),
            ));
        };

        let prompt = extract_prompt(request)
            .ok_or_else(|| ModelError::InvalidRequest("image prompt must not be empty".into()))?;

        Ok(Self {
            model: request.model.clone(),
            prompt,
            resolution: opts.resolution.clone(),
            aspect_ratio: opts.aspect_ratio.clone(),
        })
    }
}

/// 把返回的图片映射为结构化的 [`MediaSource`]。
pub fn generated_image_to_source(image: GeneratedImage) -> Option<MediaSource> {
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

#[async_trait]
impl GenerateCapability for OpenRouterImages {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError> {
        let body = ImageRequest::try_from(&request)?;
        let response = self.create_image(&body).await?;
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
            model: Some(request.model),
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
