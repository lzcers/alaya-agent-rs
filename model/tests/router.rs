use std::sync::Arc;

use alaya_model::capability::{
    FinishReason, GenerateCapability, GenerateRequest, GenerateResponse, GenerateStream,
    ModelError, ModelProfile,
};
use alaya_model::endpoints::{deepseek, openrouter};
use alaya_model::router::ModelRouter;
use alaya_model::{ContentPart, MediaSource, Message, Modality};
use async_trait::async_trait;
use futures::stream;

struct StubModel {
    tag: &'static str,
}

#[async_trait]
impl GenerateCapability for StubModel {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError> {
        let message = if request.config.wants_output(Modality::Image) {
            Message::assistant_parts(vec![ContentPart::image(MediaSource::url(format!(
                "{}:{}",
                self.tag, request.model
            )))])
        } else {
            Message::assistant(format!("{}:{}", self.tag, request.model))
        };
        Ok(GenerateResponse {
            message,
            finish_reason: Some(FinishReason::Stop),
            usage: None,
            model: Some(request.model),
            id: None,
        })
    }

    async fn generate_stream(
        &self,
        _request: GenerateRequest,
    ) -> Result<GenerateStream, ModelError> {
        Ok(Box::pin(stream::empty()))
    }
}

#[tokio::test]
async fn routes_by_model_name_and_modality() {
    let mut router = ModelRouter::new();
    router.register(
        "shared-model",
        ModelProfile::text(),
        Arc::new(StubModel { tag: "text" }),
    );
    router.register(
        "shared-model",
        ModelProfile::image_generation(),
        Arc::new(StubModel { tag: "image" }),
    );

    assert!(router.supports_output("shared-model", Modality::Text));
    assert!(router.supports_output("shared-model", Modality::Image));
    assert!(!router.supports_output("shared-model", Modality::Audio));

    let text_resp = router
        .generate(GenerateRequest::new(
            "shared-model",
            vec![Message::user("hi")],
        ))
        .await
        .unwrap();
    assert_eq!(text_resp.message, Message::assistant("text:shared-model"));

    let img_resp = router
        .generate(GenerateRequest::image("shared-model", "prompt"))
        .await
        .unwrap();
    assert_eq!(
        img_resp.message.images(),
        vec![&MediaSource::url("image:shared-model")]
    );
}

#[tokio::test]
async fn unregistered_model_reports_model_not_found() {
    let router = ModelRouter::new();

    let error = router
        .generate(GenerateRequest::new("nope", vec![]))
        .await
        .expect_err("unregistered model should fail");

    assert!(matches!(error, ModelError::ModelNotFound { model } if model == "nope"));
}

#[test]
fn registration_is_keyed_by_model_and_profile() {
    let mut router = ModelRouter::new();
    router.register_many(
        &["deepseek-chat", "deepseek-reasoner"],
        ModelProfile::text(),
        Arc::new(deepseek("dummy_key")),
    );
    router.register(
        "black-forest-labs/flux.2-klein-4b",
        ModelProfile::image_generation(),
        Arc::new(openrouter("dummy_key")),
    );

    assert!(router.supports_output("deepseek-chat", Modality::Text));
    assert!(router.supports_output("deepseek-reasoner", Modality::Text));
    assert!(!router.supports_output("black-forest-labs/flux.2-klein-4b", Modality::Text));
    assert!(router.supports_output("black-forest-labs/flux.2-klein-4b", Modality::Image));
    assert!(!router.supports_output("deepseek-chat", Modality::Image));
}

#[tokio::test]
async fn alias_rewrites_upstream_model_name() {
    let mut router = ModelRouter::new();
    router.register_alias(
        "fast",
        "openai/gpt-4o-mini",
        ModelProfile::vision(),
        Arc::new(StubModel { tag: "upstream" }),
    );

    let resp = router
        .generate(GenerateRequest::new("fast", vec![Message::user("hi")]))
        .await
        .unwrap();

    assert_eq!(
        resp.message,
        Message::assistant("upstream:openai/gpt-4o-mini")
    );
}
