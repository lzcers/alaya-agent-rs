//! 厂商身份：一个端点就是「传输 + 地址 + 请求头」。
//!
//! [`Provider`] 是**值**不是 trait——厂商之间变化的东西几乎全是数据
//! （`base_url`、密钥、额外请求头），没有第二种实现需要替换。
//! 把它做成 trait 会让「接入新厂商」从传参数退化成写代码。
//!
//! 它自己不实现任何能力；能力由 [`crate::protocols`] 的适配器在它之上实现。

use std::sync::Arc;

use futures::stream::BoxStream;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::info;

use crate::transport::{HttpRequest, HttpResponse, Transport, TransportError, sse};

/// SSE 事件的 `data` 载荷流。
pub type SseStream = BoxStream<'static, Result<String, TransportError>>;

/// 一个厂商端点。
///
/// 持有三样东西：怎么发（[`Transport`]）、发给谁（`base_url`）、怎么证明身份
/// （`headers`）。它不知道路径、wire 形状和业务语义。
#[derive(Clone)]
pub struct Provider {
    transport: Arc<dyn Transport>,
    name: String,
    base_url: String,
    headers: HeaderMap,
}

impl Provider {
    /// `name` 只用于日志与错误信息。
    pub fn new(
        transport: Arc<dyn Transport>,
        name: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        Self {
            transport,
            name: name.into(),
            base_url: base_url.into(),
            headers,
        }
    }

    /// 用 `Authorization: Bearer` 认证（RFC 6750，HTTP 标准方案）。
    ///
    /// 这是 OpenAI / OpenRouter / DeepSeek / Groq 等厂商的共同约定，
    /// 但**不是**传输层的假设——用别的方案就换 [`Provider::with_header`]。
    pub fn with_bearer(mut self, api_key: impl AsRef<str>) -> Self {
        let value = HeaderValue::from_str(&format!("Bearer {}", api_key.as_ref()))
            .expect("API key must be a valid header value");
        self.headers.insert(AUTHORIZATION, value);
        self
    }

    /// 追加一个请求头（厂商私有头，或非 Bearer 的认证方案）。
    ///
    /// # Panics
    /// header 名或值非法时立即 panic——这属于启动期配置错误，早失败好过晚 401。
    pub fn with_header(mut self, name: &str, value: impl AsRef<str>) -> Self {
        let name =
            HeaderName::try_from(name).unwrap_or_else(|_| panic!("invalid header name: {name}"));
        let value = HeaderValue::from_str(value.as_ref())
            .unwrap_or_else(|_| panic!("invalid value for header {name}"));
        self.headers.insert(name, value);
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    pub fn transport(&self) -> &Arc<dyn Transport> {
        &self.transport
    }

    /// 拼出一个指向该端点的请求（已带厂商请求头）。
    ///
    /// 这是唯一的通用出口：二进制上传、multipart、非 JSON 响应都从这里出发，
    /// 不需要给传输层或本类型加新方法。
    pub fn request(&self, method: reqwest::Method, path: &str) -> HttpRequest {
        let mut request = HttpRequest::new(method, self.url(path));
        request.headers = self.headers.clone();
        request
    }

    /// 发一个请求。
    pub async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        info!(
            provider = %self.name,
            method = %request.method,
            url = %request.url,
            "sending provider request"
        );
        self.transport.send(request).await
    }

    /// POST 一个 JSON body，返回 JSON body。
    pub async fn post_json<B, R>(&self, path: &str, body: &B) -> Result<R, TransportError>
    where
        B: Serialize + ?Sized,
        R: DeserializeOwned,
    {
        let request = self
            .request(reqwest::Method::POST, path)
            .with_json_body(body)?;
        let response = self.send(request).await?;

        ensure_success(response).await?.json().await
    }

    /// POST 一个 JSON body，返回 SSE 事件的 `data` 载荷流。
    pub async fn post_sse<B>(&self, path: &str, body: &B) -> Result<SseStream, TransportError>
    where
        B: Serialize + ?Sized,
    {
        let request = self
            .request(reqwest::Method::POST, path)
            .with_json_body(body)?
            .accepting_sse()?;
        let response = self.send(request).await?;

        let response = ensure_success(response).await?;
        Ok(Box::pin(sse::data_events(response.into_body())))
    }

    fn url(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }
}

/// 非 2xx 在这里变成错误；错误消息从常见信封里尽力提取。
async fn ensure_success(response: HttpResponse) -> Result<HttpResponse, TransportError> {
    if response.is_success() {
        return Ok(response);
    }

    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    Err(TransportError::Api {
        status,
        message: extract_error_message(&body),
    })
}

/// 尽力从错误响应体里取出一条人类可读的消息。
///
/// 不绑定任何具体厂商的信封格式：`error.message`、`message`、`error` 字符串，
/// 都没有就原样返回。
fn extract_error_message(body: &str) -> String {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return body.to_string();
    };

    value["error"]["message"]
        .as_str()
        .or_else(|| value["message"].as_str())
        .or_else(|| value["error"].as_str())
        .unwrap_or(body)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use futures::StreamExt;
    use reqwest::Method;
    use reqwest::header::ACCEPT;
    use serde_json::json;
    use std::sync::Mutex;

    #[derive(Debug, Clone)]
    struct Seen {
        method: Method,
        url: String,
        headers: HeaderMap,
        body: Bytes,
    }

    #[derive(Default)]
    struct ScriptedTransport {
        seen: Mutex<Vec<Seen>>,
        status: u16,
        body: String,
    }

    impl ScriptedTransport {
        fn responding(status: u16, body: Value) -> Arc<Self> {
            Arc::new(Self {
                status,
                body: body.to_string(),
                ..Default::default()
            })
        }

        fn seen(&self) -> Vec<Seen> {
            self.seen.lock().unwrap().clone()
        }
    }

    #[async_trait::async_trait]
    impl Transport for ScriptedTransport {
        async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
            self.seen.lock().unwrap().push(Seen {
                method: request.method.clone(),
                url: request.url.clone(),
                headers: request.headers.clone(),
                body: request.body.clone(),
            });
            Ok(HttpResponse::buffered(self.status, self.body.clone()))
        }
    }

    fn provider(transport: Arc<ScriptedTransport>) -> Provider {
        Provider::new(transport, "acme", "https://api.acme.test/v1/").with_bearer("secret")
    }

    #[tokio::test]
    async fn post_json_joins_url_injects_headers_and_decodes() {
        let transport = ScriptedTransport::responding(200, json!({ "ok": true }));
        let provider = provider(transport.clone()).with_header("x-acme-tenant", "t1");

        let value: Value = provider
            .post_json("/generate", &json!({ "prompt": "hi" }))
            .await
            .unwrap();

        assert_eq!(value["ok"], true);
        let seen = transport.seen();
        assert_eq!(seen[0].method, Method::POST);
        assert_eq!(seen[0].url, "https://api.acme.test/v1/generate");
        assert_eq!(seen[0].headers.get(AUTHORIZATION).unwrap(), "Bearer secret");
        assert_eq!(seen[0].headers.get("x-acme-tenant").unwrap(), "t1");
        assert_eq!(
            seen[0].headers.get(CONTENT_TYPE).unwrap(),
            "application/json"
        );
        assert_eq!(
            serde_json::from_slice::<Value>(&seen[0].body).unwrap(),
            json!({ "prompt": "hi" })
        );
    }

    #[tokio::test]
    async fn non_success_becomes_api_error_with_extracted_message() {
        let transport = ScriptedTransport::responding(
            401,
            json!({ "error": { "code": 401, "message": "bad key" } }),
        );
        let provider = provider(transport);

        let error = provider
            .post_json::<_, Value>("/generate", &json!({}))
            .await
            .expect_err("401 should fail");

        assert!(matches!(
            error,
            TransportError::Api { status: 401, message } if message == "bad key"
        ));
    }

    #[tokio::test]
    async fn error_message_falls_back_to_raw_body() {
        let transport = Arc::new(ScriptedTransport {
            status: 500,
            body: "upstream exploded".to_string(),
            ..Default::default()
        });

        let error = provider(transport)
            .post_json::<_, Value>("/x", &json!({}))
            .await
            .expect_err("500 should fail");

        assert!(matches!(
            error,
            TransportError::Api { status: 500, message } if message == "upstream exploded"
        ));
    }

    #[tokio::test]
    async fn a_provider_without_bearer_sends_no_authorization_header() {
        let transport = ScriptedTransport::responding(200, json!({}));
        let provider = Provider::new(
            transport.clone(),
            "anthropic",
            "https://api.anthropic.com/v1",
        )
        .with_header("x-api-key", "secret");

        let _: Value = provider.post_json("/messages", &json!({})).await.unwrap();

        let seen = transport.seen();
        assert!(seen[0].headers.get(AUTHORIZATION).is_none());
        assert_eq!(seen[0].headers.get("x-api-key").unwrap(), "secret");
    }

    #[tokio::test]
    async fn post_sse_frames_events_from_the_response_body() {
        let transport = Arc::new(ScriptedTransport {
            status: 200,
            body: "data: {\"i\":1}\n\ndata: [DONE]\n\n".to_string(),
            ..Default::default()
        });
        let provider = provider(transport.clone());

        let events = provider
            .post_sse("/stream", &json!({ "stream": true }))
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(events, vec!["{\"i\":1}", "[DONE]"]);
        assert_eq!(
            transport.seen()[0].headers.get(ACCEPT).unwrap(),
            "text/event-stream"
        );
    }
}
