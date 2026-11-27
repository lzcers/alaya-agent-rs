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
