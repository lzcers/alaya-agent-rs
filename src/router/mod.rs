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
/// may use different providers for different operations. Model selection and
/// invocation options are supplied by each request; the router keeps no call state.
#[derive(Clone)]
pub struct ModelRouter {
    routes: HashMap<ModelCapability, HashMap<String, Arc<dyn Provider>>>,
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

    pub fn supports(&self, model_name: &str, capability: ModelCapability) -> bool {
        self.routes
            .get(&capability)
            .is_some_and(|routes| routes.contains_key(model_name))
    }

    pub(super) fn route(
        &self,
        model_name: &str,
        capability: ModelCapability,
    ) -> Result<&Arc<dyn Provider>, RouterError> {
        if !self.contains_model(model_name) {
            return Err(RouterError::ModelNotFound(model_name.to_owned()));
        }
        let provider = self
            .routes
            .get(&capability)
            .and_then(|routes| routes.get(model_name))
            .ok_or_else(|| RouterError::UnsupportedCapability {
                model: model_name.to_string(),
                capability: capability.as_str(),
            })?;

        Ok(provider)
    }

    fn contains_model(&self, model_name: &str) -> bool {
        self.routes
            .values()
            .any(|routes| routes.contains_key(model_name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{deepseek_provider, openrouter_provider};

    #[test]
    fn registration_is_keyed_by_model_and_capability() {
        let provider = Arc::new(openrouter_provider("dummy_key"));
        let mut router = ModelRouter::new();
        router.add_model_provider(
            "omni-model",
            provider,
            &[ModelCapability::Chat, ModelCapability::Image],
        );

        assert!(router.route("omni-model", ModelCapability::Chat).is_ok());
        assert!(router.route("omni-model", ModelCapability::Image).is_ok());
        assert!(matches!(
            router.route("omni-model", ModelCapability::Audio),
            Err(RouterError::UnsupportedCapability { .. })
        ));
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

        let error = match router.route("deepseek-chat", ModelCapability::Image) {
            Ok(_) => panic!("image route should be rejected"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            RouterError::UnsupportedCapability {
                model,
                capability: "image"
            } if model == "deepseek-chat"
        ));
    }
}
