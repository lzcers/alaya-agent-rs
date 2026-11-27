//! 端点预设：把「哪个 `base_url` + 哪些请求头 + 认证」组装成可注册的 [`GenerateCapability`] 实现。
//!
//! 这里是组合根，没有协议知识（协议在 [`crate::protocols`]），也没有字节搬运
//! （传输在 [`crate::transport`]）——只有厂商的地址、认证与协议选择。

use std::env;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;

use crate::capability::{
    GenerateCapability, GenerateRequest, GenerateResponse, GenerateStream, ModelError,
};
use crate::protocols::{OpenAiCompatible, OpenRouterImages};
use crate::providers::Provider;
use crate::transport::{HttpTransport, Transport, TransportError};

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

/// 声明一个 OpenAI 兼容的厂商端点。
///
/// 接入一个兼容 OpenAI 的新厂商，只需要这一行：给个名字、key 和 `base_url`。
/// **不需要实现任何 trait**——协议复用 [`OpenAiCompatible`]，传输复用 [`HttpTransport`]。
pub fn openai_compatible(
    name: impl Into<String>,
    api_key: impl Into<String>,
    base_url: impl Into<String>,
) -> OpenAiCompatible {
    let provider = Provider::new(shared_transport(), name, base_url).with_bearer(api_key.into());
    OpenAiCompatible::new(provider)
}

// ============================================================================
// DeepSeek
// ============================================================================

/// DeepSeek 端点。
pub fn deepseek(api_key: impl Into<String>) -> OpenAiCompatible {
    let base_url =
        env::var("DEEPSEEK_BASE_URL").unwrap_or_else(|_| "https://api.deepseek.com".to_string());
    openai_compatible("deepseek", api_key, base_url)
}

/// 从环境变量创建 DeepSeek 端点。
///
/// 环境变量: DEEPSEEK_API_KEY (必需), DEEPSEEK_BASE_URL (可选)
pub fn deepseek_from_env() -> Result<OpenAiCompatible, TransportError> {
    let api_key = env::var("DEEPSEEK_API_KEY").map_err(|_| TransportError::MissingApiKey)?;
    Ok(deepseek(api_key))
}

// ============================================================================
// OpenRouter
// ============================================================================

/// OpenRouter 端点。
///
/// 文本/视觉/音频/多模态对话走 OpenAI 兼容协议，纯图片生成请求走 OpenRouter 自有的
/// `/images` 端点；两个适配器共享同一个 [`Provider`]，对外暴露为同一个 [`GenerateCapability`]。
pub struct OpenRouter {
    chat: OpenAiCompatible,
    images: OpenRouterImages,
}

impl OpenRouter {
    pub fn new(provider: Provider) -> Self {
        Self {
            chat: OpenAiCompatible::new(provider.clone()),
            images: OpenRouterImages::new(provider),
        }
    }

    pub fn chat_adapter(&self) -> &OpenAiCompatible {
        &self.chat
    }

    pub fn images_adapter(&self) -> &OpenRouterImages {
        &self.images
    }

    fn is_dedicated_image_request(request: &GenerateRequest) -> bool {
        request.config.is_pure_image()
    }
}

/// 创建 OpenRouter 端点。
pub fn openrouter(api_key: impl Into<String>) -> OpenRouter {
    openrouter_with_config(api_key, None, None)
}

/// 创建 OpenRouter 端点（带额外请求头配置）。
pub fn openrouter_with_config(
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

/// 从环境变量创建 OpenRouter 端点。
pub fn openrouter_from_env() -> Result<OpenRouter, TransportError> {
    let api_key = env::var("OPENROUTER_API_KEY").map_err(|_| TransportError::MissingApiKey)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;
    use crate::message::MediaSource;
    use crate::transport::{HttpRequest, HttpResponse};
    use bytes::Bytes;
    use serde_json::json;
    use std::sync::Mutex;

    #[test]
    fn deepseek_is_a_bare_openai_compatible_adapter() {
        let model = deepseek("dummy-key");

        assert_eq!(model.provider().name(), "deepseek");
        assert_eq!(model.provider().base_url(), "https://api.deepseek.com");
    }

    #[test]
    fn openrouter_is_composed_from_one_provider() {
        let model = openrouter("dummy-key");

        assert_eq!(model.chat.provider().name(), "openrouter");
        assert_eq!(model.images.provider().name(), "openrouter");
        assert_eq!(
            model.images.provider().base_url(),
            model.chat.provider().base_url()
        );
    }

    #[test]
    fn openrouter_config_headers_land_on_the_provider() {
        let model = openrouter_with_config(
            "dummy-key",
            Some("https://example.com".to_string()),
            Some("alaya".to_string()),
        );

        assert_eq!(
            model.chat.provider().header("HTTP-Referer"),
            Some("https://example.com")
        );
        assert_eq!(model.chat.provider().header("X-Title"), Some("alaya"));
    }

    #[derive(Default)]
    struct RoutingTransport {
        paths: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Transport for RoutingTransport {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
            self.paths.lock().unwrap().push(
                request
                    .url
                    .trim_start_matches("https://fake.test")
                    .to_string(),
            );

            let body = if request.url.ends_with("/images") {
                json!({ "data": [{ "b64_json": "aW1hZ2U=", "media_type": "image/png" }] })
            } else {
                json!({
                    "id": "resp_1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "m",
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": "hello" },
                        "finish_reason": "stop"
                    }]
                })
            };

            Ok(HttpResponse::buffered(200, Bytes::from(body.to_string())))
        }
    }

    #[tokio::test]
    async fn openrouter_routes_each_modality_to_its_protocol() {
        let transport = Arc::new(RoutingTransport::default());
        let provider =
            Provider::new(transport.clone(), "openrouter", "https://fake.test").with_bearer("k");
        let model = OpenRouter::new(provider);

        let text_resp = model
            .generate(GenerateRequest::new("m", vec![Message::user("hi")]))
            .await
            .unwrap();
        let img_resp = model
            .generate(GenerateRequest::image(
                "krea/krea-2-medium-turbo",
                "landscape",
            ))
            .await
            .unwrap();

        assert_eq!(text_resp.message, Message::assistant("hello"));
        assert_eq!(
            img_resp.message.images(),
            vec![&MediaSource::base64("image/png", "aW1hZ2U=")]
        );
        assert_eq!(
            *transport.paths.lock().unwrap(),
            vec!["/chat/completions".to_string(), "/images".to_string()]
        );
    }
}
