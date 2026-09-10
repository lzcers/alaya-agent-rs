//! 端点预设：把「哪个 `base_url` + 哪些请求头」组装成可注册的能力实现。
//!
//! 这里是组合根，没有协议知识（协议在 [`crate::protocols`]），也没有字节搬运
//! （传输在 [`crate::transport`]）——只有厂商的地址、认证与协议选择。
//!
//! 注意端点与协议不是一一对应的：DeepSeek 的 chat 与 OpenRouter 的 chat
//! 都是 OpenAI 兼容协议，而 OpenRouter 的图片端点走它自有的协议。

use std::env;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::Message;
use crate::capability::{
    ChatCapability, ChatChunk, ChatRequest, GenAudioCapability, GenAudioRequest, GenAudioResponse,
    GenImgCapability, GenImgRequest, GenImgResponse, ModelError,
};
use crate::protocols::{OpenAiCompatible, OpenRouterImages};
use crate::providers::Provider;
use crate::transport::{HttpTransport, Transport, TransportError};

/// 目录函数共用的默认传输。
///
/// 一个 `reqwest::Client` 就是一组连接池，跨厂商共享它是推荐做法。
/// 需要自定义超时/代理时，自己构造 `HttpTransport` 并直接组装 [`Provider`]
/// ——目录函数只是便利，不是必经之路。
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
/// **不需要实现任何 trait**——协议复用 [`OpenAiCompatible`]，传输复用
/// [`HttpTransport`]，能力声明由后续的注册方法承担。
///
/// # Example
/// ```no_run
/// use alaya_agent::{endpoints::openai_compatible, router::ModelRouter};
/// use std::sync::Arc;
///
/// let model = Arc::new(openai_compatible(
///     "groq",
///     "your-api-key",
///     "https://api.groq.com/openai/v1",
/// ));
///
/// let mut router = ModelRouter::new();
/// router.add_chat_model("llama-3.3-70b-versatile", model.clone());
/// router.add_audio_model("some-audio-model", model);
/// ```
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
///
/// 它没有任何协议分叉，所以直接返回 OpenAI 兼容协议适配器，不再套一层壳。
///
/// # Example
/// ```no_run
/// use alaya_agent::endpoints::deepseek;
/// let model = deepseek("your-api-key");
/// ```
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
/// chat / audio 走 OpenAI 兼容协议，图片走 OpenRouter 自有的统一图片 API，
/// 因此这里组合两个协议适配器，并让它们共享同一个 [`Provider`]。
pub struct OpenRouter {
    chat: OpenAiCompatible,
    images: OpenRouterImages,
}

impl OpenRouter {
    /// 用一个厂商身份组装两种协议。
    pub fn new(provider: Provider) -> Self {
        Self {
            chat: OpenAiCompatible::new(provider.clone()),
            images: OpenRouterImages::new(provider),
        }
    }
}

/// 创建 OpenRouter 端点
///
/// # Example
/// ```no_run
/// use alaya_agent::endpoints::openrouter;
/// let model = openrouter("your-api-key");
/// ```
pub fn openrouter(api_key: impl Into<String>) -> OpenRouter {
    openrouter_with_config(api_key, None, None)
}

/// 创建 OpenRouter 端点（带额外配置）
///
/// # Arguments
/// * `api_key` - API 密钥
/// * `http_referer` - HTTP-Referer 请求头（可选）
/// * `x_title` - X-Title 请求头（可选）
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

/// 从环境变量创建 OpenRouter 端点
///
/// 环境变量:
/// - OPENROUTER_API_KEY (必需)
/// - OPENROUTER_BASE_URL (可选)
/// - OPENROUTER_HTTP_REFERER (可选)
/// - OPENROUTER_X_TITLE (可选)
pub fn openrouter_from_env() -> Result<OpenRouter, TransportError> {
    let api_key = env::var("OPENROUTER_API_KEY").map_err(|_| TransportError::MissingApiKey)?;
    let http_referer = env::var("OPENROUTER_HTTP_REFERER").ok();
    let x_title = env::var("OPENROUTER_X_TITLE").ok();
    Ok(openrouter_with_config(api_key, http_referer, x_title))
}

#[async_trait]
impl ChatCapability for OpenRouter {
    async fn chat(&self, request: ChatRequest) -> Result<Message, ModelError> {
        self.chat.chat(request).await
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, ChatChunk>, ModelError> {
        self.chat.chat_stream(request).await
    }
}

#[async_trait]
impl GenAudioCapability for OpenRouter {
    async fn gen_audio(&self, request: GenAudioRequest) -> Result<GenAudioResponse, ModelError> {
        self.chat.gen_audio(request).await
    }
}

#[async_trait]
impl GenImgCapability for OpenRouter {
    async fn gen_img(&self, request: GenImgRequest) -> Result<GenImgResponse, ModelError> {
        self.images.gen_img(request).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let headers = model.chat.provider().headers();
        assert_eq!(headers.get("HTTP-Referer").unwrap(), "https://example.com");
        assert_eq!(headers.get("X-Title").unwrap(), "alaya");
    }

    /// 记录路径并按路径返回不同响应的传输，用来验证端点如何把能力分发到不同协议。
    #[derive(Default)]
    struct RoutingTransport {
        paths: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Transport for RoutingTransport {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
            self.paths
                .lock()
                .unwrap()
                .push(request.url.trim_start_matches("https://fake.test").to_string());

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
    async fn openrouter_routes_each_capability_to_its_protocol() {
        let transport = Arc::new(RoutingTransport::default());
        let provider =
            Provider::new(transport.clone(), "openrouter", "https://fake.test").with_bearer("k");
        let model = OpenRouter::new(provider);

        let message = model
            .chat(ChatRequest::new("m", vec![Message::user("hi")]))
            .await
            .unwrap();
        let image = model
            .gen_img(GenImgRequest::new("krea/krea-2-medium-turbo", "landscape"))
            .await
            .unwrap();

        assert_eq!(message, Message::assistant("hello"));
        assert_eq!(image.image_urls, vec!["data:image/png;base64,aW1hZ2U="]);
        assert_eq!(
            *transport.paths.lock().unwrap(),
            vec!["/chat/completions".to_string(), "/images".to_string()]
        );
    }
}
