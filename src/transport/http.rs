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

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use serde_json::json;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    #[test]
    fn transport_holds_connection_config_only() {
        let transport = HttpTransport::new()
            .with_proxy("http://127.0.0.1:7890")
            .with_timeout(Duration::from_secs(60));

        assert_eq!(transport.proxy_url(), Some("http://127.0.0.1:7890"));
        assert_eq!(transport.timeout(), Duration::from_secs(60));
    }

    #[tokio::test]
    async fn send_posts_bytes_and_returns_buffered_json() {
        let (base_url, server) =
            serve_once("HTTP/1.1 200 OK", &json!({ "ok": true }).to_string()).await;
        let transport = HttpTransport::new();

        let response = transport
            .send(
                HttpRequest::post(format!("{base_url}/chat/completions"))
                    .with_header("x-test", "1")
                    .unwrap()
                    .with_json_body(&json!({ "model": "m" }))
                    .unwrap(),
            )
            .await
            .expect("request should succeed");

        let raw = server.await.expect("test server should complete");
        let (headers, body) = raw.split_once("\r\n\r\n").expect("headers and body");

        assert!(headers.starts_with("POST /chat/completions HTTP/1.1"));
        assert!(headers.to_ascii_lowercase().contains("x-test: 1"));
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(body).unwrap(),
            json!({ "model": "m" })
        );
        assert!(response.is_success());
        assert_eq!(
            response.json::<serde_json::Value>().await.unwrap()["ok"],
            true
        );
    }

    #[tokio::test]
    async fn send_returns_raw_bytes_for_binary_responses() {
        let (base_url, server) = serve_once("HTTP/1.1 200 OK", "not-json-at-all").await;
        let transport = HttpTransport::new();

        let response = transport
            .send(HttpRequest::post(format!("{base_url}/audio/speech")))
            .await
            .expect("request should succeed");
        let _ = server.await;

        assert_eq!(
            response.bytes().await.unwrap(),
            Bytes::from_static(b"not-json-at-all")
        );
    }

    #[tokio::test]
    async fn send_does_not_fail_on_non_success_status() {
        let (base_url, server) = serve_once("HTTP/1.1 429 Too Many Requests", "slow down").await;
        let transport = HttpTransport::new();

        let response = transport
            .send(HttpRequest::post(format!("{base_url}/x")))
            .await
            .expect("transport should return the response, not an error");
        let _ = server.await;

        assert_eq!(response.status(), 429);
        assert!(!response.is_success());
    }

    /// 起一个只服务一次请求的 HTTP 服务端，返回 (base_url, 请求原文句柄)。
    async fn serve_once(
        status_line: &str,
        response_body: &str,
    ) -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener should bind");
        let address = listener.local_addr().expect("listener address");
        let status_line = status_line.to_string();
        let response = format!(
            "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
            response_body.len()
        );

        let handle = tokio::spawn(async move {
            let (mut socket, _) = listener
                .accept()
                .await
                .expect("test server should accept a connection");
            let mut request = Vec::new();
            let mut expected_len = None;

            loop {
                let mut chunk = [0_u8; 4096];
                let bytes_read = socket
                    .read(&mut chunk)
                    .await
                    .expect("test server should read request");
                if bytes_read == 0 {
                    break;
                }
                request.extend_from_slice(&chunk[..bytes_read]);

                if expected_len.is_none()
                    && let Some(header_end) =
                        request.windows(4).position(|part| part == b"\r\n\r\n")
                {
                    let headers = String::from_utf8_lossy(&request[..header_end]);
                    let content_length = headers
                        .lines()
                        .find_map(|line| {
                            let (name, value) = line.split_once(':')?;
                            name.eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or_default();
                    expected_len = Some(header_end + 4 + content_length);
                }

                if expected_len.is_some_and(|length| request.len() >= length) {
                    break;
                }
            }

            socket
                .write_all(response.as_bytes())
                .await
                .expect("test server should write response");

            String::from_utf8(request).expect("request should be UTF-8")
        });

        (format!("http://{address}"), handle)
    }
}
