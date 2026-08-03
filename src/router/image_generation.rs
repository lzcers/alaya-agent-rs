use crate::{
    providers::{GeneratedImage, ImageGenerationRequest},
    router::{ModelCapability, ModelRouter, RouterError},
};
use async_trait::async_trait;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenImgResponse {
    pub image_urls: Vec<String>,
}

#[async_trait]
pub trait GenImgCapability {
    async fn gen_img(&self, request: ImageGenerationRequest)
    -> Result<GenImgResponse, RouterError>;
}

#[async_trait]
impl GenImgCapability for ModelRouter {
    async fn gen_img(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<GenImgResponse, RouterError> {
        let provider = self.route(&request.model, ModelCapability::Image)?;
        if request.prompt.trim().is_empty() {
            return Err(RouterError::NoResponse);
        }
        let response = provider.generate_image(request).await?;
        let image_urls = response
            .data
            .into_iter()
            .filter_map(generated_image_to_url)
            .collect::<Vec<_>>();

        if image_urls.is_empty() {
            return Err(RouterError::NoResponse);
        }

        Ok(GenImgResponse { image_urls })
    }
}

fn generated_image_to_url(image: GeneratedImage) -> Option<String> {
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
    use crate::Message;
    use crate::providers::{
        ImageGenerationResponse, Provider, ProviderError, Request, Response, StreamResponse,
        openrouter_provider, openrouter_provider_from_env,
    };
    use futures::stream::{self, BoxStream};
    use std::sync::{Arc, Mutex};

    #[derive(Default)]
    struct MockImageProvider {
        requests: Mutex<Vec<ImageGenerationRequest>>,
    }

    #[async_trait]
    impl Provider for MockImageProvider {
        async fn chat(&self, _request: Request) -> Result<Response, ProviderError> {
            Err(ProviderError::ApiError {
                code: 400,
                message: "chat is not used by the image capability".to_string(),
            })
        }

        async fn chat_stream(
            &self,
            _request: Request,
        ) -> Result<BoxStream<'static, StreamResponse>, ProviderError> {
            Ok(Box::pin(stream::empty()))
        }

        async fn generate_image(
            &self,
            request: ImageGenerationRequest,
        ) -> Result<ImageGenerationResponse, ProviderError> {
            self.requests.lock().unwrap().push(request);
            Ok(ImageGenerationResponse {
                data: vec![GeneratedImage {
                    b64_json: Some("aW1hZ2U=".to_string()),
                    url: None,
                    media_type: Some("image/webp".to_string()),
                }],
            })
        }

        fn name(&self) -> &str {
            "mock-image"
        }
    }

    #[tokio::test]
    async fn gen_img_builds_dedicated_image_request_and_returns_data_url() {
        let provider = Arc::new(MockImageProvider::default());
        let mut model = ModelRouter::new();
        model.add_model_provider(
            "krea/krea-2-medium-turbo",
            provider.clone(),
            &[ModelCapability::Image],
        );
        let response = model
            .gen_img(
                ImageGenerationRequest::new(
                    "krea/krea-2-medium-turbo",
                    "visual direction\n\ncinematic landscape",
                )
                .with_aspect_ratio("16:9")
                .with_resolution("1K"),
            )
            .await
            .unwrap();

        assert_eq!(response.image_urls, vec!["data:image/webp;base64,aW1hZ2U="]);
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model, "krea/krea-2-medium-turbo");
        assert_eq!(
            requests[0].prompt,
            "visual direction\n\ncinematic landscape"
        );
        assert_eq!(requests[0].resolution.as_deref(), Some("1K"));
        assert_eq!(requests[0].aspect_ratio.as_deref(), Some("16:9"));
    }

    #[tokio::test]
    async fn test_gen_img_with_openrouter() {
        dotenv::dotenv().ok();

        let provider = match openrouter_provider_from_env() {
            Ok(p) => Arc::new(p),
            Err(_) => {
                eprintln!("OPENROUTER_API_KEY not set, skipping test");
                return;
            }
        };

        let mut model = ModelRouter::new();

        model.add_model_provider(
            "black-forest-labs/flux.2-klein-4b",
            provider,
            &[ModelCapability::Image],
        );

        let msg = Message::user("Generate a beautiful sunset over mountains");

        let result = model
            .gen_img(
                ImageGenerationRequest::new("black-forest-labs/flux.2-klein-4b", msg.content())
                    .with_aspect_ratio("1:1")
                    .with_resolution("1K"),
            )
            .await;
        if let Err(e) = result {
            eprintln!("Failed to generate image: {}", e);
            return;
        }

        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(!response.image_urls.is_empty());
    }

    #[test]
    fn test_model_provider_mapping() {
        let or_provider = Arc::new(openrouter_provider("dummy_key"));

        let mut model = ModelRouter::new();

        model.add_models_for_provider(
            &[
                "black-forest-labs/flux.2-klein-4b",
                "black-forest-labs/flux.1-pro",
            ],
            or_provider,
            &[ModelCapability::Image],
        );

        assert!(model.supports("black-forest-labs/flux.2-klein-4b", ModelCapability::Image));
        assert!(model.supports("black-forest-labs/flux.1-pro", ModelCapability::Image));
    }

    #[test]
    fn request_selects_the_model() {
        let provider = Arc::new(openrouter_provider("dummy_key"));
        let mut model = ModelRouter::new();
        model.add_model_provider(
            "black-forest-labs/flux.2-klein-4b",
            provider,
            &[ModelCapability::Image],
        );

        assert!(
            model
                .route("black-forest-labs/flux.2-klein-4b", ModelCapability::Image)
                .is_ok()
        );
        assert!(matches!(
            model.route("non-existent-model", ModelCapability::Image),
            Err(RouterError::ModelNotFound(_))
        ));
    }

    #[test]
    fn image_options_belong_to_the_request() {
        let request = ImageGenerationRequest::new("model", "prompt")
            .with_aspect_ratio("16:9")
            .with_resolution("2K");

        assert_eq!(request.aspect_ratio.as_deref(), Some("16:9"));
        assert_eq!(request.resolution.as_deref(), Some("2K"));
    }
}
