//! HTTP 传输实现。
//!
//! `Transport` 是这一层唯一的抽象：**发一次请求**。它不知道厂商、密钥和业务语义——
//! URL、请求头、请求体都是调用方给的参数。

use async_trait::async_trait;
use futures::StreamExt;
use std::time::Duration;
use tracing::debug;

use super::{HttpRequest, HttpResponse, TransportError};

/// 传输抽象。
///
/// 只认识字节：请求由 [`HttpRequest`] 完整描述，响应以 [`HttpResponse`] 返回，
/// 响应体永远是字节流。是否缓冲、怎么解释（JSON / SSE / 裸音频）由调用方决定，
/// 所以新增二进制接口不需要给本 trait 加方法。
///
/// 实现者只需要关心「怎么把这段字节发到这个地址」。
#[async_trait]
pub trait Transport: Send + Sync {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError>;
}

/// 基于 reqwest 的传输实现。
///
/// 只持有连接配置：连接池、超时、代理。没有 base_url，没有密钥，没有厂商名。
#[derive(Clone)]
pub struct HttpTransport {
    client: reqwest::Client,
    timeout: Duration,
    proxy_url: Option<String>,
}

impl Default for HttpTransport {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpTransport {
    pub fn new() -> Self {
        let timeout = Duration::from_secs(120);
        Self {
            client: build_http_client(timeout, None),
            timeout,
            proxy_url: None,
        }
    }

    /// 设置请求超时时间
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self.client = build_http_client(self.timeout, self.proxy_url.as_deref());
        self
    }

    /// 设置请求代理
    pub fn with_proxy(mut self, proxy_url: impl Into<String>) -> Self {
        self.proxy_url = Some(proxy_url.into());
        self.client = build_http_client(self.timeout, self.proxy_url.as_deref());
        self
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    pub fn proxy_url(&self) -> Option<&str> {
        self.proxy_url.as_deref()
    }
}

fn build_http_client(timeout: Duration, proxy_url: Option<&str>) -> reqwest::Client {
    let mut builder = reqwest::Client::builder().timeout(timeout);

    if let Some(proxy_url) = proxy_url.map(str::trim).filter(|url| !url.is_empty()) {
        builder = builder.proxy(reqwest::Proxy::all(proxy_url).expect("Invalid proxy URL"));
    }

    builder.build().expect("Failed to build HTTP client")
}

#[async_trait]
impl Transport for HttpTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        debug!(
            method = %request.method,
            url = %request.url,
            proxy_configured = self.proxy_url.is_some(),
            "sending http request"
        );

        let response = self
            .client
            .request(request.method, &request.url)
            .headers(request.headers)
            .body(request.body)
            .send()
            .await?;

        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let body = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(TransportError::from));

        Ok(HttpResponse::new(status, headers, Box::pin(body)))
    }
}
