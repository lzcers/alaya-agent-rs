//! 端点预设：把「`base_url` + 请求头 + 认证 + 协议」组装成可直接使用的 [`GenerateCapability`]。
//!
//! 对上层只暴露统一的能力句柄 `Arc<dyn GenerateCapability>`，隐藏底层协议适配器与传输细节。

use std::env;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;

use crate::capability::{
    GenerateCapability, GenerateRequest, GenerateResponse, GenerateStream, ModelError,
};
use crate::protocols::{OpenAiCompatible, OpenRouterImages};
use crate::providers::Provider;
use crate::transport::{HttpTransport, Transport};

/// 目录函数共用的默认传输。
fn shared_transport() -> Arc<dyn Transport> {
    static SHARED: OnceLock<Arc<dyn Transport>> = OnceLock::new();
    SHARED
        .get_or_init(|| Arc::new(HttpTransport::new()))
        .clone()
}

// ============================================================================
// 通用：任何 OpenAI 兼容厂商
// ============================================================================

pub(crate) fn build_openai_compatible(
    name: impl Into<String>,
    api_key: impl Into<String>,
    base_url: impl Into<String>,
) -> OpenAiCompatible {
    let provider = Provider::new(shared_transport(), name, base_url).with_bearer(api_key.into());
    OpenAiCompatible::new(provider)
}

/// 声明一个 OpenAI 兼容的厂商端点。
///
/// 接入兼容 OpenAI 的新厂商只需要名字、API Key 和 `base_url`。
pub fn openai_compatible(
    name: impl Into<String>,
    api_key: impl Into<String>,
    base_url: impl Into<String>,
) -> Arc<dyn GenerateCapability> {
    Arc::new(build_openai_compatible(name, api_key, base_url))
}

// ============================================================================
// DeepSeek
// ============================================================================

pub(crate) fn build_deepseek(api_key: impl Into<String>) -> OpenAiCompatible {
    let base_url =
        env::var("DEEPSEEK_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com".to_string());
    build_openai_compatible("deepseek", api_key, base_url)
}

/// DeepSeek 端点。
pub fn deepseek(api_key: impl Into<String>) -> Arc<dyn GenerateCapability> {
    Arc::new(build_deepseek(api_key))
}

/// 从环境变量创建 DeepSeek 端点。
///
/// 环境变量: `DEEPSEEK_API_KEY` (必需), `DEEPSEEK_BASE_URL` (可选)
pub fn deepseek_from_env() -> Result<Arc<dyn GenerateCapability>, ModelError> {
    let api_key = env::var("DEEPSEEK_API_KEY").map_err(|_| ModelError::MissingApiKey)?;
    Ok(deepseek(api_key))
}

// ============================================================================
// OpenRouter
// ============================================================================

/// OpenRouter 端点。
///
/// 文本/视觉/音频/多模态对话走 OpenAI 兼容协议，纯图片生成请求走 OpenRouter 自有的
/// `/images` 端点；两个适配器共享同一个 [`Provider`]，对外暴露为同一个 [`GenerateCapability`]。
pub(crate) struct OpenRouter {
    chat: OpenAiCompatible,
    images: OpenRouterImages,
}

impl OpenRouter {
    pub(crate) fn new(provider: Provider) -> Self {
        Self {
            chat: OpenAiCompatible::new(provider.clone()),
            images: OpenRouterImages::new(provider),
        }
    }

    #[cfg(test)]
    pub(crate) fn chat_adapter(&self) -> &OpenAiCompatible {
        &self.chat
    }

    #[cfg(test)]
    pub(crate) fn images_adapter(&self) -> &OpenRouterImages {
        &self.images
    }

    fn is_dedicated_image_request(request: &GenerateRequest) -> bool {
        request.config.is_pure_image()
    }
}

pub(crate) fn build_openrouter(
    api_key: impl Into<String>,
    http_referer: Option<String>,
    x_title: Option<String>,
) -> OpenRouter {
    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let mut provider =
        Provider::new(shared_transport(), "openrouter", base_url).with_bearer(api_key.into());
    if let Some(referer) = http_referer {
        provider = provider.with_header("HTTP-Referer", referer);
    }
    if let Some(title) = x_title {
        provider = provider.with_header("X-Title", title);
    }

    OpenRouter::new(provider)
}

/// 创建 OpenRouter 端点。
pub fn openrouter(api_key: impl Into<String>) -> Arc<dyn GenerateCapability> {
    openrouter_with_config(api_key, None, None)
}

/// 创建 OpenRouter 端点（带额外请求头配置）。
pub fn openrouter_with_config(
    api_key: impl Into<String>,
    http_referer: Option<String>,
    x_title: Option<String>,
) -> Arc<dyn GenerateCapability> {
    Arc::new(build_openrouter(api_key, http_referer, x_title))
}

/// 从环境变量创建 OpenRouter 端点。
///
/// 环境变量: `OPENROUTER_API_KEY` (必需), `OPENROUTER_BASE_URL` / `OPENROUTER_HTTP_REFERER` / `OPENROUTER_X_TITLE` (可选)
pub fn openrouter_from_env() -> Result<Arc<dyn GenerateCapability>, ModelError> {
    let api_key = env::var("OPENROUTER_API_KEY").map_err(|_| ModelError::MissingApiKey)?;
    let http_referer = env::var("OPENROUTER_HTTP_REFERER").ok();
    let x_title = env::var("OPENROUTER_X_TITLE").ok();
    Ok(openrouter_with_config(api_key, http_referer, x_title))
}

#[async_trait]
impl GenerateCapability for OpenRouter {
    async fn generate(&self, request: GenerateRequest) -> Result<GenerateResponse, ModelError> {
        if Self::is_dedicated_image_request(&request) {
            self.images.generate(request).await
        } else {
            self.chat.generate(request).await
        }
    }

    async fn generate_stream(
        &self,
        request: GenerateRequest,
    ) -> Result<GenerateStream, ModelError> {
        if Self::is_dedicated_image_request(&request) {
            self.images.generate_stream(request).await
        } else {
            self.chat.generate_stream(request).await
        }
    }
}
