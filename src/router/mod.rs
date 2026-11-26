//! 分发层：按逻辑模型名与能力画像把统一生成请求路由到具体协议适配器。
//!
//! - 一张路由表统一承载所有模态（无需按模态拆分多个 `HashMap`）
//! - 支持逻辑模型名 → 上游模型名映射（解决跨厂商同名冲突与 Azure deployment 映射）
//! - 调用前按 [`ModelProfile`] 校验输入/输出模态与工具能力
//! - 同一逻辑名下注册多条路由时，按模态筛选并在遇到可重试错误时顺序切换主备

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::capability::{
    GenerateCapability, GenerateRequest, GenerateResponse, GenerateStream, ModelError, ModelProfile,
};
use crate::message::Modality;

/// 一条模型路由目标。
#[derive(Clone)]
pub struct ModelRoute {
    /// 发往上游协议的实际模型名（可与逻辑路由键不同）
    pub upstream_model: String,
    pub profile: ModelProfile,
    pub adapter: Arc<dyn GenerateCapability>,
}

impl ModelRoute {
    pub fn new(
        upstream_model: impl Into<String>,
        profile: ModelProfile,
        adapter: Arc<dyn GenerateCapability>,
    ) -> Self {
        Self {
            upstream_model: upstream_model.into(),
            profile,
            adapter,
        }
    }
}

/// 统一的多模态模型路由器。
#[derive(Clone, Default)]
pub struct ModelRouter {
    routes: HashMap<String, Vec<ModelRoute>>,
}

impl ModelRouter {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册一条路由（逻辑名与上游模型名相同）。
    pub fn register(
        &mut self,
        model: &str,
        profile: ModelProfile,
        adapter: Arc<dyn GenerateCapability>,
    ) {
        self.register_route(model, ModelRoute::new(model, profile, adapter));
    }

    /// 批量注册共享同一能力画像和适配器的模型。
    pub fn register_many(
        &mut self,
        models: &[&str],
        profile: ModelProfile,
        adapter: Arc<dyn GenerateCapability>,
    ) {
        for model in models {
            self.register(model, profile.clone(), adapter.clone());
        }
    }

    /// 注册逻辑别名，将 `logical_model` 映射到上游的 `upstream_model`。
    pub fn register_alias(
        &mut self,
        logical_model: &str,
        upstream_model: &str,
        profile: ModelProfile,
        adapter: Arc<dyn GenerateCapability>,
    ) {
        self.register_route(
            logical_model,
            ModelRoute::new(upstream_model, profile, adapter),
        );
    }

    /// 追加一条路由到指定逻辑模型名下（多条兼容路由按注册顺序充当主备）。
    pub fn register_route(&mut self, logical_model: &str, route: ModelRoute) {
        self.routes
            .entry(logical_model.to_owned())
            .or_default()
            .push(route);
    }

    pub fn contains(&self, model: &str) -> bool {
        self.routes.contains_key(model)
    }

    pub fn routes(&self, model: &str) -> Option<&[ModelRoute]> {
        self.routes.get(model).map(Vec::as_slice)
    }

    pub fn supports_input(&self, model: &str, modality: Modality) -> bool {
        self.routes.get(model).is_some_and(|list| {
            list.iter()
                .any(|route| route.profile.input_modalities.contains(&modality))
        })
    }

    pub fn supports_output(&self, model: &str, modality: Modality) -> bool {
        self.routes.get(model).is_some_and(|list| {
            list.iter()
                .any(|route| route.profile.output_modalities.contains(&modality))
        })
    }

    fn resolve<'a>(&'a self, request: &GenerateRequest) -> Result<Vec<&'a ModelRoute>, ModelError> {
        let Some(candidates) = self.routes.get(&request.model) else {
            return Err(ModelError::ModelNotFound {
                model: request.model.clone(),
            });
        };

        let mut matched = Vec::new();
        let mut first_mismatch = None;

        for route in candidates {
            match route.profile.validate(request) {
                Ok(()) => matched.push(route),
                Err(err) => {
                    if first_mismatch.is_none() {
                        first_mismatch = Some(err);
                    }
                }
            }
        }

        if matched.is_empty() {
            return Err(first_mismatch.unwrap_or_else(|| ModelError::ModelNotFound {
                model: request.model.clone(),
            }));
        }

        Ok(matched)
    }
}

#[async_trait]
impl GenerateCapability for ModelRouter {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError> {
        let candidates = self.resolve(&request)?;
        let last_index = candidates.len() - 1;

        for (index, route) in candidates.into_iter().enumerate() {
            let mut upstream_request = request.clone();
            upstream_request.model = route.upstream_model.clone();
            match route.adapter.generate(upstream_request).await {
                Ok(response) => return Ok(response),
                Err(error) if error.is_retryable() && index < last_index => continue,
                Err(error) => return Err(error),
            }
        }

        unreachable!("candidates is non-empty")
    }

    async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateStream, ModelError> {
        let candidates = self.resolve(&request)?;
        let last_index = candidates.len() - 1;

        for (index, route) in candidates.into_iter().enumerate() {
            let mut upstream_request = request.clone();
            upstream_request.model = route.upstream_model.clone();
            match route.adapter.generate_stream(upstream_request).await {
                Ok(stream) => return Ok(stream),
                Err(error) if error.is_retryable() && index < last_index => continue,
                Err(error) => return Err(error),
            }
        }

        unreachable!("candidates is non-empty")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;
    use crate::capability::FinishReason;
    use crate::endpoints::{deepseek, openrouter};
    use crate::message::{ContentPart, MediaSource};
    use futures::stream;

    struct StubModel {
        tag: &'static str,
    }

    #[async_trait]
    impl GenerateCapability for StubModel {
        async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError> {
            let message = if request.output_modalities.contains(&Modality::Image) {
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
        assert!(!router.supports_output(
            "black-forest-labs/flux.2-klein-4b",
            Modality::Text
        ));
        assert!(router.supports_output(
            "black-forest-labs/flux.2-klein-4b",
            Modality::Image
        ));
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
}
