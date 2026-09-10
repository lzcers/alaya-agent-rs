//! 分发层：把模型名映射到已注册的能力实现。
//!
//! 这里**不做**任何协议映射——只查表、转发、把「没注册」翻译成
//! [`ModelError::ModelNotFound`]。协议映射在 [`crate::protocols`]。

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::Message;
use crate::capability::{
    ChatCapability, ChatChunk, ChatRequest, GenAudioCapability, GenAudioRequest, GenAudioResponse,
    GenImgCapability, GenImgRequest, GenImgResponse, ModelError,
};

/// 按 (能力, 模型名) 分发的路由器。
///
/// 能力由**注册时调用的方法**决定，而不是运行时枚举：调用
/// [`ModelRouter::add_image_model`] 就是在断言「这个模型能生成图片」。
/// 同一模型名可以在不同能力上指向不同实现。
#[derive(Clone, Default)]
pub struct ModelRouter {
    chat: HashMap<String, Arc<dyn ChatCapability>>,
    image: HashMap<String, Arc<dyn GenImgCapability>>,
    audio: HashMap<String, Arc<dyn GenAudioCapability>>,
}

impl ModelRouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册 chat 能力。同名模型已注册时保留原实现。
    pub fn add_chat_model(&mut self, model: &str, chat: Arc<dyn ChatCapability>) {
        self.chat.entry(model.to_owned()).or_insert(chat);
    }

    pub fn add_chat_models(&mut self, models: &[&str], chat: Arc<dyn ChatCapability>) {
        for model in models {
            self.add_chat_model(model, chat.clone());
        }
    }

    pub fn add_image_model(&mut self, model: &str, image: Arc<dyn GenImgCapability>) {
        self.image.entry(model.to_owned()).or_insert(image);
    }

    pub fn add_image_models(&mut self, models: &[&str], image: Arc<dyn GenImgCapability>) {
        for model in models {
            self.add_image_model(model, image.clone());
        }
    }

    pub fn add_audio_model(&mut self, model: &str, audio: Arc<dyn GenAudioCapability>) {
        self.audio.entry(model.to_owned()).or_insert(audio);
    }

    pub fn add_audio_models(&mut self, models: &[&str], audio: Arc<dyn GenAudioCapability>) {
        for model in models {
            self.add_audio_model(model, audio.clone());
        }
    }

    pub fn supports_chat(&self, model: &str) -> bool {
        self.chat.contains_key(model)
    }

    pub fn supports_image(&self, model: &str) -> bool {
        self.image.contains_key(model)
    }

    pub fn supports_audio(&self, model: &str) -> bool {
        self.audio.contains_key(model)
    }

    fn not_found(model: &str, capability: &'static str) -> ModelError {
        ModelError::ModelNotFound {
            model: model.to_owned(),
            capability,
        }
    }
}

#[async_trait]
impl ChatCapability for ModelRouter {
    async fn chat(&self, request: ChatRequest) -> Result<Message, ModelError> {
        let model = self
            .chat
            .get(&request.model)
            .ok_or_else(|| Self::not_found(&request.model, "chat"))?;

        model.chat(request).await
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, ChatChunk>, ModelError> {
        let model = self
            .chat
            .get(&request.model)
            .ok_or_else(|| Self::not_found(&request.model, "chat"))?;

        model.chat_stream(request).await
    }
}

#[async_trait]
impl GenImgCapability for ModelRouter {
    async fn gen_img(&self, request: GenImgRequest) -> Result<GenImgResponse, ModelError> {
        let model = self
            .image
            .get(&request.model)
            .ok_or_else(|| Self::not_found(&request.model, "image"))?;

        model.gen_img(request).await
    }
}

#[async_trait]
impl GenAudioCapability for ModelRouter {
    async fn gen_audio(&self, request: GenAudioRequest) -> Result<GenAudioResponse, ModelError> {
        let model = self
            .audio
            .get(&request.model)
            .ok_or_else(|| Self::not_found(&request.model, "audio"))?;

        model.gen_audio(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;
    use crate::capability::ChatChunk;
    use crate::endpoints::{deepseek, openrouter};
    use futures::stream;

    struct StubChat;

    #[async_trait]
    impl ChatCapability for StubChat {
        async fn chat(&self, request: ChatRequest) -> Result<Message, ModelError> {
            Ok(Message::assistant(format!("chat:{}", request.model)))
        }

        async fn chat_stream(
            &self,
            _request: ChatRequest,
        ) -> Result<BoxStream<'static, ChatChunk>, ModelError> {
            Ok(Box::pin(stream::empty()))
        }
    }

    struct StubImage;

    #[async_trait]
    impl GenImgCapability for StubImage {
        async fn gen_img(&self, _request: GenImgRequest) -> Result<GenImgResponse, ModelError> {
            Ok(GenImgResponse {
                image_urls: vec!["stub".to_string()],
            })
        }
    }

    #[tokio::test]
    async fn routes_by_model_name_and_capability() {
        let mut router = ModelRouter::new();
        router.add_chat_model("shared-model", Arc::new(StubChat));
        router.add_image_model("shared-model", Arc::new(StubImage));

        assert!(router.supports_chat("shared-model"));
        assert!(router.supports_image("shared-model"));
        assert!(!router.supports_audio("shared-model"));

        let message = router
            .chat(ChatRequest::new("shared-model", vec![Message::user("hi")]))
            .await
            .unwrap();
        assert_eq!(message, Message::assistant("chat:shared-model"));

        let image = router
            .gen_img(GenImgRequest::new("shared-model", "prompt"))
            .await
            .unwrap();
        assert_eq!(image.image_urls, vec!["stub".to_string()]);
    }

    #[tokio::test]
    async fn unregistered_model_reports_capability_and_name() {
        let router = ModelRouter::new();

        let error = router
            .chat(ChatRequest::new("nope", vec![]))
            .await
            .expect_err("unregistered model should fail");

        assert!(matches!(
            error,
            ModelError::ModelNotFound { model, capability }
                if model == "nope" && capability == "chat"
        ));
    }

    #[test]
    fn registration_is_keyed_by_model_and_capability() {
        let mut router = ModelRouter::new();
        router.add_chat_models(
            &["deepseek-chat", "deepseek-reasoner"],
            Arc::new(deepseek("dummy_key")),
        );
        router.add_image_models(
            &["black-forest-labs/flux.2-klein-4b"],
            Arc::new(openrouter("dummy_key")),
        );

        assert!(router.supports_chat("deepseek-chat"));
        assert!(router.supports_chat("deepseek-reasoner"));
        assert!(!router.supports_chat("black-forest-labs/flux.2-klein-4b"));
        assert!(router.supports_image("black-forest-labs/flux.2-klein-4b"));
        assert!(!router.supports_image("deepseek-chat"));
    }

    #[test]
    fn first_registration_wins() {
        let mut router = ModelRouter::new();
        router.add_chat_model("m", Arc::new(StubChat));
        router.add_chat_model("m", Arc::new(StubChat));

        assert_eq!(router.chat.len(), 1);
    }
}
