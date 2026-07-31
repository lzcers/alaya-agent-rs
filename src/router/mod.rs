mod audio_generation;
mod chat_completion;
mod error;
mod image_generation;

pub use audio_generation::{GenAudioCapability, GenAudioResponse};
pub use chat_completion::{ChatCapability, ChatChunk};
pub use error::{RouterError, format_router_error};
pub use image_generation::{GenImgCapability, GenImgResponse};

use std::{collections::HashMap, fmt, sync::Arc};

use crate::providers::Provider;

/// A model operation supported by a route.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelCapability {
    Chat,
    Image,
    Audio,
}

impl ModelCapability {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Image => "image",
            Self::Audio => "audio",
        }
    }
}

impl fmt::Display for ModelCapability {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Routes model operations to their provider.
///
/// Registration is keyed by capability and model name, so the same model name
/// may use different providers for different operations. Each capability also
/// keeps an independent active model and its own request options.
#[derive(Clone)]
pub struct ModelRouter {
    routes: HashMap<ModelCapability, HashMap<String, Arc<dyn Provider>>>,
    active_models: HashMap<ModelCapability, String>,

    output_json: bool,
    reasoning_effort: Option<String>,
    thinking_enabled: Option<bool>,

    aspect_ratio: String,
    image_size: String,

    audio_format: String,
    voice: Option<String>,
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelRouter {
    pub fn new() -> Self {
        Self {
            routes: HashMap::new(),
            active_models: HashMap::new(),
            output_json: false,
            reasoning_effort: None,
            thinking_enabled: None,
            aspect_ratio: "1:1".to_string(),
            image_size: "1K".to_string(),
            audio_format: "wav".to_string(),
            voice: None,
        }
    }

    /// Registers a model and the operations it supports.
    ///
    /// If a `(capability, model_name)` route is already registered, its
    /// original provider is retained.
    pub fn add_model_provider(
        &mut self,
        model_name: &str,
        provider: Arc<dyn Provider>,
        capabilities: &[ModelCapability],
    ) {
        for capability in capabilities {
            self.routes
                .entry(*capability)
                .or_default()
                .entry(model_name.to_owned())
                .or_insert_with(|| provider.clone());
        }
    }

    pub fn add_models_for_provider(
        &mut self,
        model_names: &[&str],
        provider: Arc<dyn Provider>,
        capabilities: &[ModelCapability],
    ) {
        for model_name in model_names {
            self.add_model_provider(model_name, provider.clone(), capabilities);
        }
    }

    pub fn set_active_model(
        &mut self,
        capability: ModelCapability,
        model_name: &str,
    ) -> Result<(), RouterError> {
        if !self.contains_model(model_name) {
            return Err(RouterError::ModelNotFound(model_name.to_owned()));
        }
        if !self.supports(model_name, capability) {
            return Err(RouterError::UnsupportedCapability {
                model: model_name.to_owned(),
                capability: capability.as_str(),
            });
        }

        self.active_models.insert(capability, model_name.to_owned());
        Ok(())
    }

    pub fn active_model(&self, capability: ModelCapability) -> Option<&str> {
        self.active_models.get(&capability).map(String::as_str)
    }

    pub fn supports(&self, model_name: &str, capability: ModelCapability) -> bool {
        self.routes
            .get(&capability)
            .is_some_and(|routes| routes.contains_key(model_name))
    }

    pub fn set_output_json(&mut self, output_json: bool) {
        self.output_json = output_json;
    }

    pub fn set_reasoning_effort(&mut self, reasoning_effort: impl Into<String>) {
        self.reasoning_effort = Some(reasoning_effort.into());
    }

    pub fn set_thinking_enabled(&mut self, enabled: bool) {
        self.thinking_enabled = Some(enabled);
    }

    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = aspect_ratio.into();
        self
    }

    pub fn with_image_size(mut self, image_size: impl Into<String>) -> Self {
        self.image_size = image_size.into();
        self
    }

    pub fn with_audio_format(mut self, audio_format: impl Into<String>) -> Self {
        self.audio_format = audio_format.into();
        self
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }

    pub(super) fn route(
        &self,
        capability: ModelCapability,
    ) -> Result<(&str, &Arc<dyn Provider>), RouterError> {
        let model_name = self
            .active_models
            .get(&capability)
            .ok_or(RouterError::NoActiveModel(capability.as_str()))?;
        let provider = self
            .routes
            .get(&capability)
            .and_then(|routes| routes.get(model_name))
            .ok_or_else(|| RouterError::UnsupportedCapability {
                model: model_name.clone(),
                capability: capability.as_str(),
            })?;

        Ok((model_name, provider))
    }

    fn contains_model(&self, model_name: &str) -> bool {
        self.routes
            .values()
            .any(|routes| routes.contains_key(model_name))
    }

    pub(super) fn output_json(&self) -> bool {
        self.output_json
    }

    pub(super) fn reasoning_effort(&self) -> Option<&str> {
        self.reasoning_effort.as_deref()
    }

    pub(super) fn thinking_enabled(&self) -> Option<bool> {
        self.thinking_enabled
    }

    pub(super) fn aspect_ratio(&self) -> &str {
        &self.aspect_ratio
    }

    pub(super) fn image_size(&self) -> &str {
        &self.image_size
    }

    pub(super) fn audio_format(&self) -> &str {
        &self.audio_format
    }

    pub(super) fn voice(&self) -> Option<&str> {
        self.voice.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{deepseek_provider, openrouter_provider};

    #[test]
    fn registration_is_shared_and_active_models_are_per_capability() {
        let provider = Arc::new(openrouter_provider("dummy_key"));
        let mut router = ModelRouter::new();
        router.add_model_provider(
            "omni-model",
            provider,
            &[ModelCapability::Chat, ModelCapability::Image],
        );

        router
            .set_active_model(ModelCapability::Chat, "omni-model")
            .unwrap();
        router
            .set_active_model(ModelCapability::Image, "omni-model")
            .unwrap();

        assert_eq!(
            router.active_model(ModelCapability::Chat),
            Some("omni-model")
        );
        assert_eq!(
            router.active_model(ModelCapability::Image),
            Some("omni-model")
        );
        assert_eq!(router.active_model(ModelCapability::Audio), None);
    }

    #[test]
    fn capabilities_can_use_different_providers_for_the_same_model_name() {
        let chat_provider = Arc::new(deepseek_provider("dummy_key"));
        let image_provider = Arc::new(openrouter_provider("dummy_key"));
        let mut router = ModelRouter::new();
        router.add_model_provider(
            "shared-model",
            chat_provider.clone(),
            &[ModelCapability::Chat],
        );
        router.add_model_provider(
            "shared-model",
            image_provider.clone(),
            &[ModelCapability::Image],
        );

        let chat_route = &router.routes[&ModelCapability::Chat]["shared-model"];
        let image_route = &router.routes[&ModelCapability::Image]["shared-model"];
        assert!(Arc::ptr_eq(
            chat_route,
            &(chat_provider as Arc<dyn Provider>)
        ));
        assert!(Arc::ptr_eq(
            image_route,
            &(image_provider as Arc<dyn Provider>)
        ));
    }

    #[test]
    fn rejects_unsupported_capability() {
        let provider = Arc::new(deepseek_provider("dummy_key"));
        let mut router = ModelRouter::new();
        router.add_model_provider("deepseek-chat", provider, &[ModelCapability::Chat]);

        let error = router
            .set_active_model(ModelCapability::Image, "deepseek-chat")
            .unwrap_err();

        assert!(matches!(
            error,
            RouterError::UnsupportedCapability {
                model,
                capability: "image"
            } if model == "deepseek-chat"
        ));
    }
}
