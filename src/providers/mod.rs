//! 厂商身份：一个端点就是「传输 + 地址 + 请求头 + 认证」。
//!
//! [`Provider`] 是**值**不是 trait——厂商之间变化的东西几乎全是数据
//! （`base_url`、认证方案、额外请求头），动态签名（如 AWS SigV4 / OAuth2 刷新）
//! 通过 [`Auth::Custom`] 注入，而不需要把整个厂商抽象成 trait。
//!
//! 它自己不实现任何能力；能力由 [`crate::protocols`] 的适配器在它之上实现。

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::stream::BoxStream;
use reqwest::header::RETRY_AFTER;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::info;

use crate::transport::{HttpRequest, HttpResponse, SseEvent, Transport, TransportError, sse};

/// SSE 事件流。
pub type SseStream = BoxStream<'static, Result<SseEvent, TransportError>>;

/// 动态请求认证/签名钩子（如 AWS SigV4、OAuth2 Token 刷新）。
#[async_trait]
pub trait RequestAuth: Send + Sync {
    async fn apply(&self, request: &mut HttpRequest) -> Result<(), TransportError>;
}

/// 厂商认证方案。
#[derive(Clone, Default)]
pub enum Auth {
    #[default]
    None,
    /// `Authorization: Bearer <token>`
    Bearer(String),
    /// 自定义请求头（如 Anthropic `x-api-key`、Azure `api-key`）
    Header { name: String, value: String },
    /// Query 参数认证（如 Gemini `?key=...`）
    QueryParam { name: String, value: String },
    /// 动态按请求签名或刷新令牌
    Custom(Arc<dyn RequestAuth>),
}

/// 一个厂商端点。
///
/// 持有四样东西：怎么发（[`Transport`]）、发给谁（`base_url`）、默认头（`headers`）、
/// 以及怎么证明身份（[`Auth`]）。它不知道路径、wire 形状和业务语义。
#[derive(Clone)]
pub struct Provider {
    transport: Arc<dyn Transport>,
    name: String,
    base_url: String,
    headers: Vec<(String, String)>,
    auth: Auth,
}

impl Provider {
    /// `name` 只用于日志与错误信息。
    pub fn new(
        transport: Arc<dyn Transport>,
        name: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            transport,
            name: name.into(),
            base_url: base_url.into(),
            headers: Vec::new(),
            auth: Auth::None,
        }
    }

    /// 用 `Authorization: Bearer` 认证（RFC 6750）。
    pub fn with_bearer(mut self, api_key: impl Into<String>) -> Self {
        self.auth = Auth::Bearer(api_key.into());
        self
    }

    /// 用自定义请求头认证（如 `x-api-key`）。
    pub fn with_header_auth(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth = Auth::Header {
            name: name.into(),
            value: value.into(),
        };
        self
    }

    /// 用 query 参数认证（如 `?key=...`）。
    pub fn with_query_auth(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth = Auth::QueryParam {
            name: name.into(),
            value: value.into(),
        };
        self
    }

    /// 设置任意认证方案（含动态签名 [`Auth::Custom`]）。
    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    /// 追加一个厂商默认请求头（在发送请求时校验，不在构造期 panic）。
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        let name = name.into();
        let value = value.into();
        if let Some((_, existing)) = self
            .headers
            .iter_mut()
            .find(|(k, _)| k.eq_ignore_ascii_case(&name))
        {
            *existing = value;
        } else {
            self.headers.push((name, value));
        }
        self
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn auth(&self) -> &Auth {
        &self.auth
    }

    pub fn transport(&self) -> &Arc<dyn Transport> {
        &self.transport
    }

    /// 拼出一个指向该端点的请求（已带厂商默认请求头，保留 `base_url` 与 `path` 的 query 参数）。
    ///
    /// 这是唯一的通用出口：二进制上传、multipart、非 JSON 响应都从这里出发。
    pub fn request(
        &self,
        method: reqwest::Method,
        path: &str,
    ) -> Result<HttpRequest, TransportError> {
        let mut request = HttpRequest::new(method, join_url(&self.base_url, path));
        for (name, value) in &self.headers {
            request = request.with_header(name, value)?;
        }
        Ok(request)
    }

    /// 应用认证并发送请求。
    pub async fn send(&self, mut request: HttpRequest) -> Result<HttpResponse, TransportError> {
        apply_auth(&self.auth, &mut request).await?;
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
            .request(reqwest::Method::POST, path)?
            .with_json_body(body)?;
        let response = self.send(request).await?;

        ensure_success(response).await?.json().await
    }

    /// POST 一个 JSON body，返回标准 SSE 事件流。
    pub async fn post_sse<B>(&self, path: &str, body: &B) -> Result<SseStream, TransportError>
    where
        B: Serialize + ?Sized,
    {
        let request = self
            .request(reqwest::Method::POST, path)?
            .with_json_body(body)?
            .accepting_sse()?;
        let response = self.send(request).await?;

        let response = ensure_success(response).await?;
        Ok(Box::pin(sse::events(response.into_body())))
    }
}

async fn apply_auth(auth: &Auth, request: &mut HttpRequest) -> Result<(), TransportError> {
    match auth {
        Auth::None => Ok(()),
        Auth::Bearer(token) => {
            let value = format!("Bearer {token}");
            let updated = std::mem::replace(request, HttpRequest::post(""))
                .with_header("authorization", value)?;
            *request = updated;
            Ok(())
        }
        Auth::Header { name, value } => {
            let updated = std::mem::replace(request, HttpRequest::post(""))
                .with_header(name, value)?;
            *request = updated;
            Ok(())
        }
        Auth::QueryParam { name, value } => {
            let updated =
                std::mem::replace(request, HttpRequest::post("")).with_query(name, value);
            *request = updated;
            Ok(())
        }
        Auth::Custom(signer) => signer.apply(request).await,
    }
}

/// 拼接 `base_url` 与相对 `path`，同时保留两边的 query 参数（支持 Azure `api-version`、Gemini `alt=sse` 等）。
fn join_url(base_url: &str, path: &str) -> String {
    let (base_path, base_query) = base_url.split_once('?').unwrap_or((base_url, ""));
    let (rel_path, rel_query) = path.split_once('?').unwrap_or((path, ""));

    let mut url = if rel_path.is_empty() {
        base_path.trim_end_matches('/').to_string()
    } else {
        format!(
            "{}/{}",
            base_path.trim_end_matches('/'),
            rel_path.trim_start_matches('/')
        )
    };

    match (base_query.is_empty(), rel_query.is_empty()) {
        (true, true) => {}
        (false, true) => {
            url.push('?');
            url.push_str(base_query);
        }
        (true, false) => {
            url.push('?');
            url.push_str(rel_query);
        }
        (false, false) => {
            url.push('?');
            url.push_str(base_query);
            url.push('&');
            url.push_str(rel_query);
        }
    }

    url
}

/// 非 2xx 在这里变成错误；错误消息从常见信封里尽力提取，同时解析 `Retry-After`。
pub async fn ensure_success(response: HttpResponse) -> Result<HttpResponse, TransportError> {
    if response.is_success() {
        return Ok(response);
    }

    let status = response.status();
    let retry_after = response
        .headers()
        .get(RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(Duration::from_secs);
    let body = response.text().await.unwrap_or_default();
    Err(TransportError::Api {
        status,
        message: extract_error_message(&body),
        retry_after,
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
    use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap};
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
            TransportError::Api { status: 401, message, .. } if message == "bad key"
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
            TransportError::Api { status: 500, message, .. } if message == "upstream exploded"
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
        .with_header_auth("x-api-key", "secret");

        let _: Value = provider.post_json("/messages", &json!({})).await.unwrap();

        let seen = transport.seen();
        assert!(seen[0].headers.get(AUTHORIZATION).is_none());
        assert_eq!(seen[0].headers.get("x-api-key").unwrap(), "secret");
    }

    #[tokio::test]
    async fn post_sse_frames_events_from_the_response_body() {
        let transport = Arc::new(ScriptedTransport {
            status: 200,
            body: "event: msg\ndata: {\"i\":1}\n\ndata: [DONE]\n\n".to_string(),
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

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event.as_deref(), Some("msg"));
        assert_eq!(events[0].data, "{\"i\":1}");
        assert_eq!(events[1].data, "[DONE]");
        assert_eq!(
            transport.seen()[0].headers.get(ACCEPT).unwrap(),
            "text/event-stream"
        );
    }
}
