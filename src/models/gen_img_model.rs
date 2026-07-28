use std::{collections::HashMap, sync::Arc};

use crate::{
    core::Message,
    models::{ChatError, GenImgCapability, GenImgResponse},
    providers::{GeneratedImage, ImageGenerationRequest, Provider},
};
use async_trait::async_trait;

pub struct GenImgModel {
    model_providers: HashMap<String, Arc<dyn Provider>>,
    active_model: Option<String>,
    aspect_ratio: String,
    image_size: String,
}

impl Default for GenImgModel {
    fn default() -> Self {
        Self::new()
    }
}

impl GenImgModel {
    pub fn new() -> Self {
        Self {
            model_providers: HashMap::new(),
            active_model: None,
            aspect_ratio: "1:1".to_string(),
            image_size: "1K".to_string(),
        }
    }

    pub fn add_model_provider(&mut self, model_name: &str, provider: Arc<dyn Provider>) {
        self.model_providers
            .entry(model_name.to_owned())
            .or_insert(provider);
    }

    pub fn add_models_for_provider(&mut self, model_names: &[&str], provider: Arc<dyn Provider>) {
        for model_name in model_names {
            self.add_model_provider(model_name, provider.clone());
        }
    }

    pub fn set_active_model(&mut self, model_name: &str) -> Result<(), ChatError> {
        if !self.model_providers.contains_key(model_name) {
            return Err(ChatError::ModelNotFound(model_name.to_owned()));
        }
        self.active_model = Some(model_name.to_owned());
        Ok(())
    }

    pub fn with_aspect_ratio(mut self, aspect_ratio: String) -> Self {
        self.aspect_ratio = aspect_ratio;
        self
    }

    pub fn with_image_size(mut self, image_size: String) -> Self {
        self.image_size = image_size;
        self
    }

    fn get_provider(&self, model_name: &str) -> Result<&Arc<dyn Provider>, ChatError> {
        self.model_providers
            .get(model_name)
            .ok_or_else(|| ChatError::ModelNotFound(model_name.to_owned()))
    }

    pub fn active_model(&self) -> Option<&str> {
        self.active_model.as_deref()
    }
}

#[async_trait]
impl GenImgCapability for GenImgModel {
    async fn gen_img(&self, msgs: Vec<Message>) -> Result<GenImgResponse, ChatError> {
        let model_name = self
            .active_model
            .as_ref()
            .ok_or_else(|| ChatError::ModelNotFound("No active model set".to_string()))?;

        let provider = self.get_provider(model_name)?;

        let prompt = msgs
            .iter()
            .map(Message::content)
            .map(str::trim)
            .filter(|content| !content.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        if prompt.is_empty() {
            return Err(ChatError::NoResponse);
        }

        let request = ImageGenerationRequest::new(model_name, prompt)
            .with_resolution(&self.image_size)
            .with_aspect_ratio(&self.aspect_ratio);
        let response = provider.generate_image(request).await?;
        let image_urls = response
            .data
            .into_iter()
            .filter_map(generated_image_to_url)
            .collect::<Vec<_>>();

        if image_urls.is_empty() {
            return Err(ChatError::NoResponse);
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
    use crate::core::Message;
    use crate::providers::{
        ImageGenerationResponse, ProviderError, Request, Response, StreamResponse,
        openrouter_provider, openrouter_provider_from_env,
    };
    use futures::stream::{self, BoxStream};
    use std::sync::Mutex;

    #[derive(Default)]
    struct MockImageProvider {
        requests: Mutex<Vec<ImageGenerationRequest>>,
    }

    #[async_trait]
    impl Provider for MockImageProvider {
        async fn chat(&self, _request: Request) -> Result<Response, ProviderError> {
            Err(ProviderError::ApiError {
                code: 400,
                message: "chat is not used by GenImgModel".to_string(),
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
        let mut model = GenImgModel::new()
            .with_aspect_ratio("16:9".to_string())
            .with_image_size("1K".to_string());
        model.add_model_provider("krea/krea-2-medium-turbo", provider.clone());
        model.set_active_model("krea/krea-2-medium-turbo").unwrap();

        let response = model
            .gen_img(vec![
                Message::system("visual direction"),
                Message::user("cinematic landscape"),
            ])
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

        let mut model = GenImgModel::new()
            .with_aspect_ratio("1:1".to_string())
            .with_image_size("1K".to_string());

        model.add_model_provider("black-forest-labs/flux.2-klein-4b", provider);

        if let Err(e) = model.set_active_model("black-forest-labs/flux.2-klein-4b") {
            eprintln!("Failed to set active model: {}", e);
            return;
        }

        let msg = Message::user("Generate a beautiful sunset over mountains");

        let result = model.gen_img(vec![msg]).await;
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

        let mut model = GenImgModel::new();

        model.add_models_for_provider(
            &[
                "black-forest-labs/flux.2-klein-4b",
                "black-forest-labs/flux.1-pro",
            ],
            or_provider,
        );

        assert!(
            model
                .model_providers
                .contains_key("black-forest-labs/flux.2-klein-4b")
        );
        assert!(
            model
                .model_providers
                .contains_key("black-forest-labs/flux.1-pro")
        );
        assert_eq!(model.model_providers.len(), 2);
    }

    #[test]
    fn test_set_active_model() {
        let provider = Arc::new(openrouter_provider("dummy_key"));
        let mut model = GenImgModel::new();
        model.add_model_provider("black-forest-labs/flux.2-klein-4b", provider);

        let result = model.set_active_model("black-forest-labs/flux.2-klein-4b");
        assert!(result.is_ok());
        assert_eq!(
            model.active_model,
            Some("black-forest-labs/flux.2-klein-4b".to_string())
        );

        let result = model.set_active_model("non-existent-model");
        assert!(result.is_err());
        assert!(matches!(result, Err(ChatError::ModelNotFound(_))));
    }

    #[test]
    fn test_builder_methods() {
        let model = GenImgModel::new()
            .with_aspect_ratio("16:9".to_string())
            .with_image_size("2K".to_string());

        assert_eq!(model.aspect_ratio, "16:9");
        assert_eq!(model.image_size, "2K");
    }
}
