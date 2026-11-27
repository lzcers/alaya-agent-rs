//! 厂商身份：一个端点就是「传输 + 地址 + 请求头 + 认证」。
//!
//! [`Provider`] 是**值**不是 trait——厂商之间变化的东西几乎全是数据
//! （`base_url`、认证方案、额外请求头）。它自己不实现任何能力；
//! 能力由 [`crate::protocols`] 的适配器在它之上实现。

use std::sync::Arc;
use std::time::Duration;

use futures::stream::BoxStream;
use reqwest::header::RETRY_AFTER;
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use tracing::info;

use crate::transport::{HttpRequest, HttpResponse, SseEvent, Transport, TransportError, sse};

/// SSE 事件流。
pub type SseStream = BoxStream<'static, Result<SseEvent, TransportError>>;

/// 厂商认证方案。
#[derive(Clone, Default)]
pub enum Auth {
    #[default]
    None,
    /// `Authorization: Bearer <token>`
    Bearer(String),
    /// 自定义请求头（如 `x-api-key`）
    #[cfg(test)]
    Header { name: String, value: String },
}

/// 一个厂商端点。
///
/// 持有四样东西：怎么发（[`Transport`]）、发给谁（`base_url`）、默认头（`headers`）、
/// 以及怎么证明身份（[`Auth`]）。它不知道路径、协议形状和业务语义。
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
    #[cfg(test)]
    pub fn with_header_auth(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.auth = Auth::Header {
            name: name.into(),
            value: value.into(),
        };
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

    #[cfg(test)]
    pub fn name(&self) -> &str {
        &self.name
    }

    #[cfg(test)]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    #[cfg(test)]
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// 拼出一个指向该端点的请求（已带厂商默认请求头，保留 `base_url` 与 `path` 的 query 参数）。
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
        apply_auth(&self.auth, &mut request)?;
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

fn apply_auth(auth: &Auth, request: &mut HttpRequest) -> Result<(), TransportError> {
    match auth {
        Auth::None => Ok(()),
        Auth::Bearer(token) => {
            let value = format!("Bearer {token}");
            let updated = std::mem::replace(request, HttpRequest::post(""))
                .with_header("authorization", value)?;
            *request = updated;
            Ok(())
        }
        #[cfg(test)]
        Auth::Header { name, value } => {
            let updated =
                std::mem::replace(request, HttpRequest::post("")).with_header(name, value)?;
            *request = updated;
            Ok(())
        }
    }
}

/// 拼接 `base_url` 与相对 `path`，同时保留两边的 query 参数。
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
async fn ensure_success(response: HttpResponse) -> Result<HttpResponse, TransportError> {
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
