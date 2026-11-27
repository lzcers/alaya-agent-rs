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
use crate::message::{ContentPart, Message, MessageRole};
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
        let Some(opts) = request.config.image_options().cloned() else {
            return Err(ModelError::Unsupported(
                "OpenRouterImages only supports `image` output modality".to_string(),
            ));
        };

        let prompt = extract_prompt(&request)
            .ok_or_else(|| ModelError::InvalidRequest("image prompt must not be empty".into()))?;

        let model = request.model.clone();
        let wire = WireImageRequest {
            model: request.model,
            prompt,
            resolution: opts.resolution,
            aspect_ratio: opts.aspect_ratio,
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
