//! 统一的多模态生成能力契约与模型能力画像。

use std::sync::Arc;

use async_trait::async_trait;

use crate::capability::{GenerateRequest, GenerateResponse, GenerateStream, ModelError};
use crate::message::Modality;

/// 模型能力画像：声明该模型支持的输入/输出模态与关键能力。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProfile {
    pub input_modalities: Vec<Modality>,
    pub output_modalities: Vec<Modality>,
    pub supports_tools: bool,
    pub supports_reasoning: bool,
    pub supports_streaming: bool,
    pub context_window: Option<u32>,
}

impl ModelProfile {
    /// 纯文本对话模型（支持工具与流式）。
    pub fn text() -> Self {
        Self {
            input_modalities: vec![Modality::Text],
            output_modalities: vec![Modality::Text],
            supports_tools: true,
            supports_reasoning: false,
            supports_streaming: true,
            context_window: None,
        }
    }

    /// 视觉语言模型（文本+图片输入 → 文本输出）。
    pub fn vision() -> Self {
        Self {
            input_modalities: vec![Modality::Text, Modality::Image],
            output_modalities: vec![Modality::Text],
            supports_tools: true,
            supports_reasoning: false,
            supports_streaming: true,
            context_window: None,
        }
    }

    /// 全模态模型（全输入模态 → 文本/图片/音频输出，支持工具与推理）。
    pub fn omni() -> Self {
        Self {
            input_modalities: vec![
                Modality::Text,
                Modality::Image,
                Modality::Audio,
                Modality::Video,
                Modality::File,
            ],
            output_modalities: vec![Modality::Text, Modality::Image, Modality::Audio],
            supports_tools: true,
            supports_reasoning: true,
            supports_streaming: true,
            context_window: None,
        }
    }

    /// 专用生图模型（文本输入 → 图片输出）。
    pub fn image_generation() -> Self {
        Self {
            input_modalities: vec![Modality::Text],
            output_modalities: vec![Modality::Image],
            supports_tools: false,
            supports_reasoning: false,
            supports_streaming: false,
            context_window: None,
        }
    }

    /// 专用音频生成模型（文本输入 → 音频输出）。
    pub fn audio_generation() -> Self {
        Self {
            input_modalities: vec![Modality::Text],
            output_modalities: vec![Modality::Audio],
            supports_tools: false,
            supports_reasoning: false,
            supports_streaming: true,
            context_window: None,
        }
    }

    pub fn with_input_modalities(mut self, modalities: impl IntoIterator<Item = Modality>) -> Self {
        self.input_modalities = modalities.into_iter().collect();
        self
    }

    pub fn with_output_modalities(
        mut self,
        modalities: impl IntoIterator<Item = Modality>,
    ) -> Self {
        self.output_modalities = modalities.into_iter().collect();
        self
    }

    pub fn with_tools(mut self, supported: bool) -> Self {
        self.supports_tools = supported;
        self
    }

    pub fn with_reasoning(mut self, supported: bool) -> Self {
        self.supports_reasoning = supported;
        self
    }

    pub fn with_streaming(mut self, supported: bool) -> Self {
        self.supports_streaming = supported;
        self
    }

    pub fn with_context_window(mut self, tokens: u32) -> Self {
        self.context_window = Some(tokens);
        self
    }

    /// 在调用上游前校验请求模态与工具需求是否满足该模型画像。
    pub fn validate(&self, request: &GenerateRequest) -> Result<(), ModelError> {
        for modality in request.input_modalities() {
            if !self.input_modalities.contains(&modality) {
                return Err(ModelError::Unsupported(format!(
                    "model `{}` does not support `{modality}` input",
                    request.model
                )));
            }
        }
        for modality in request.output_modalities() {
            if !self.output_modalities.contains(&modality) {
                return Err(ModelError::Unsupported(format!(
                    "model `{}` does not support `{modality}` output",
                    request.model
                )));
            }
        }
        if request.tools.as_ref().is_some_and(|t| !t.is_empty()) && !self.supports_tools {
            return Err(ModelError::Unsupported(format!(
                "model `{}` does not support tools",
                request.model
            )));
        }
        Ok(())
    }
}

/// 统一的多模态生成能力 trait。
///
/// 由协议适配器（[`crate::protocols`]）、端点组合体（[`crate::endpoints`]）
/// 与路由器（[`crate::router::ModelRouter`]）实现。
#[async_trait]
pub trait GenerateCapability: Send + Sync {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError>;

    async fn generate_stream(&self, request: GenerateRequest)
    -> Result<GenerateStream, ModelError>;
}

#[async_trait]
impl<T: GenerateCapability + ?Sized> GenerateCapability for Arc<T> {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError> {
        (**self).generate(request).await
    }

    async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateStream, ModelError> {
        (**self).generate_stream(request).await
    }
}
