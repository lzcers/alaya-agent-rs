//! 端点预设：把「哪个 `base_url` + 哪些默认请求头」组装成可注册的能力实现。
//!
//! 这里没有协议知识（协议在 [`crate::protocols`]），也没有 HTTP 细节
//! （传输在 [`crate::providers`]）——只有厂商的地址与命名。
//!
//! 注意端点与协议不是一一对应的：DeepSeek 的 chat 与 OpenRouter 的 chat
//! 都是 OpenAI 兼容协议，而 OpenRouter 的图片端点走它自有的协议。

use std::collections::HashMap;
use std::env;
use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::Message;
use crate::capability::{
    ChatCapability, ChatChunk, ChatRequest, GenAudioCapability, GenAudioRequest, GenAudioResponse,
    GenImgCapability, GenImgRequest, GenImgResponse, ModelError,
};
use crate::protocols::{OpenAiCompatible, OpenRouterImages};
use crate::providers::{HttpProvider, Provider, ProviderError};

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
    OpenAiCompatible::new(Arc::new(HttpProvider::new("deepseek", api_key, base_url)))
}

/// 从环境变量创建 DeepSeek 端点。
///
/// 环境变量: DEEPSEEK_API_KEY (必需), DEEPSEEK_BASE_URL (可选)
pub fn deepseek_from_env() -> Result<OpenAiCompatible, ProviderError> {
    let api_key = env::var("DEEPSEEK_API_KEY").map_err(|_| ProviderError::MissingApiKey)?;
    Ok(deepseek(api_key))
}

// ============================================================================
// OpenRouter
// ============================================================================

/// OpenRouter 端点。
///
/// chat / audio 走 OpenAI 兼容协议，图片走 OpenRouter 自有的统一图片 API，
/// 因此这里组合两个协议适配器，并让它们共享同一个传输实例。
pub struct OpenRouter {
    chat: OpenAiCompatible,
    images: OpenRouterImages,
}

impl OpenRouter {
    /// 用同一个传输实例组装两种协议（共享连接池、认证与代理配置）。
    pub fn new(http: Arc<dyn Provider>) -> Self {
        Self {
            chat: OpenAiCompatible::new(http.clone()),
            images: OpenRouterImages::new(http),
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
    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());
    OpenRouter::new(Arc::new(HttpProvider::new("openrouter", api_key, base_url)))
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
    let mut extra = HashMap::new();
    if let Some(r) = http_referer {
        extra.insert("HTTP-Referer".into(), r);
    }
    if let Some(t) = x_title {
        extra.insert("X-Title".into(), t);
    }

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    OpenRouter::new(Arc::new(
        HttpProvider::new("openrouter", api_key, base_url).with_extra_headers(extra),
    ))
}

/// 从环境变量创建 OpenRouter 端点
///
/// 环境变量:
/// - OPENROUTER_API_KEY (必需)
/// - OPENROUTER_BASE_URL (可选)
/// - OPENROUTER_HTTP_REFERER (可选)
/// - OPENROUTER_X_TITLE (可选)
pub fn openrouter_from_env() -> Result<OpenRouter, ProviderError> {
    let api_key = env::var("OPENROUTER_API_KEY").map_err(|_| ProviderError::MissingApiKey)?;
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
    use serde_json::{Value, json};
    use std::sync::Mutex;

    #[test]
    fn deepseek_is_a_bare_openai_compatible_adapter() {
        let model = deepseek("dummy-key");

        assert_eq!(model.http().name(), "deepseek");
    }

    /// 记录路径并按路径返回不同响应的传输，用来验证端点如何把能力分发到不同协议。
    #[derive(Default)]
    struct RoutingTransport {
        paths: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl Provider for RoutingTransport {
        async fn post_json(&self, path: &str, _body: Value) -> Result<Value, ProviderError> {
            self.paths.lock().unwrap().push(path.to_string());
            if path == "/images" {
                Ok(json!({ "data": [{ "b64_json": "aW1hZ2U=", "media_type": "image/png" }] }))
            } else {
                Ok(json!({
                    "id": "resp_1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "m",
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": "hello" },
                        "finish_reason": "stop"
                    }]
                }))
            }
        }

        async fn post_sse(
            &self,
            _path: &str,
            _body: Value,
        ) -> Result<BoxStream<'static, Result<String, ProviderError>>, ProviderError> {
            unreachable!("chat_stream is not exercised in this test")
        }

        fn name(&self) -> &str {
            "routing"
        }
    }

    #[tokio::test]
    async fn openrouter_routes_each_capability_to_its_protocol() {
        let transport = Arc::new(RoutingTransport::default());
        let model = OpenRouter::new(transport.clone());

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
