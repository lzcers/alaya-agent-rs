use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap};
use serde_json::Value;
use std::collections::HashMap;
use std::time::Duration;
use tracing::info;

use super::{ProviderError, parse_api_error, sse::drain_events};

/// HTTP 传输抽象。
///
/// 这一层只认识「协议动词」：发一个 JSON 请求拿一个 JSON 响应，
/// 或者发一个 JSON 请求拿一串 SSE 事件载荷。
///
/// 它刻意不认识任何业务操作——没有 `chat`、`generate_image` 之类的方法。
/// 端点路径、wire 结构体、以及 wire ↔ 域类型的映射都属于协议层。
/// 这样新增一种能力（embeddings、rerank、TTS……）不需要改动本 trait
/// 及其任何实现，也不需要给测试桩补桩。
#[async_trait]
pub trait Provider: Send + Sync {
    /// POST 一个 JSON body，返回 JSON body。
    async fn post_json(&self, path: &str, body: Value) -> Result<Value, ProviderError>;

    /// POST 一个 JSON body，返回 SSE 事件的 `data` 载荷流。
    ///
    /// 载荷不做任何解释；流中途的读取失败以 `Err` 形式出现在流里。
    async fn post_sse(
        &self,
        path: &str,
        body: Value,
    ) -> Result<BoxStream<'static, Result<String, ProviderError>>, ProviderError>;

    /// 传输标签，仅用于日志与错误信息，不参与任何逻辑分支。
    fn name(&self) -> &str;
}

/// 基于 reqwest 的 HTTP 传输实现。
///
/// 负责 URL 拼接、认证头、超时、代理与 SSE 分帧；不关心请求/响应的业务含义。
pub struct HttpProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    extra_headers: HashMap<String, String>,
    name: String,
    timeout: Duration,
    proxy_url: Option<String>,
}

impl HttpProvider {
    /// 创建新的 HTTP 传输。
    ///
    /// # Arguments
    /// * `name` - 日志标签（如 "deepseek"）
    /// * `api_key` - API 密钥
    /// * `base_url` - API 基础 URL（如 "https://api.deepseek.com"）
    pub fn new(
        name: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        let timeout = Duration::from_secs(120);

        Self {
            client: build_http_client(timeout, None),
            api_key: api_key.into(),
            base_url: base_url.into(),
            extra_headers: HashMap::new(),
            name: name.into(),
            timeout,
            proxy_url: None,
        }
    }

    /// 添加额外的请求头
    pub fn with_extra_headers(mut self, headers: HashMap<String, String>) -> Self {
        self.extra_headers = headers;
        self
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

    /// 设置自定义基础 URL
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    fn url(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    fn build_headers(&self, accept_sse: bool) -> HeaderMap {
        use reqwest::header::HeaderName;

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", self.api_key)
                .parse()
                .expect("Invalid API key format"),
        );
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        if accept_sse {
            headers.insert(ACCEPT, "text/event-stream".parse().unwrap());
        }

        for (key, value) in &self.extra_headers {
            if let Ok(name) = HeaderName::try_from(key.as_str())
                && let Ok(val) = value.parse()
            {
                headers.insert(name, val);
            }
        }

        headers
    }

    fn log_request(&self, path: &str, url: &str) {
        let proxy_url = self.proxy_url.as_deref().unwrap_or("<none>");
        info!(
            provider = %self.name,
            path = %path,
            url = %url,
            proxy_configured = self.proxy_url.is_some(),
            proxy_url = %proxy_url,
            "sending provider request"
        );
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
impl Provider for HttpProvider {
    async fn post_json(&self, path: &str, body: Value) -> Result<Value, ProviderError> {
        let url = self.url(path);
        self.log_request(path, &url);

        let response = self
            .client
            .post(&url)
            .headers(self.build_headers(false))
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(parse_api_error(&body, status.as_u16()));
        }

        Ok(serde_json::from_str(&body)?)
    }

    async fn post_sse(
        &self,
        path: &str,
        body: Value,
    ) -> Result<BoxStream<'static, Result<String, ProviderError>>, ProviderError> {
        let url = self.url(path);
        self.log_request(path, &url);

        let response = self
            .client
            .post(&url)
            .headers(self.build_headers(true))
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(parse_api_error(&body, status.as_u16()));
        }

        let mut bytes = response.bytes_stream();
        let mut buffer = Vec::new();

        let events = async_stream::try_stream! {
            while let Some(chunk) = bytes.next().await {
                let chunk =
                    chunk.map_err(|e| ProviderError::StreamError(e.to_string()))?;
                buffer.extend_from_slice(&chunk);
                for event in drain_events(&mut buffer) {
                    yield event;
                }
            }

            // 有些服务端最后一条事件不带结尾空行，补一个终止符把残留冲刷出来。
            buffer.extend_from_slice(b"\n\n");
            for event in drain_events(&mut buffer) {
                yield event;
            }
        };

        Ok(Box::pin(events))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    #[test]
    fn test_provider_creation() {
        let provider = HttpProvider::new("test", "test-api-key", "https://api.example.com");

        assert_eq!(provider.name(), "test");
        assert_eq!(provider.api_key, "test-api-key");
        assert_eq!(provider.base_url, "https://api.example.com");
    }

    #[test]
    fn url_joins_base_and_path_without_duplicating_slashes() {
        let provider = HttpProvider::new("test", "key", "https://api.example.com/v1/");

        assert_eq!(
            provider.url("/chat/completions"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_with_extra_headers() {
        let mut extra = HashMap::new();
        extra.insert("X-Custom-Header".to_string(), "custom-value".to_string());

        let provider = HttpProvider::new("test", "test-api-key", "https://api.example.com")
            .with_extra_headers(extra);

        assert_eq!(provider.extra_headers.len(), 1);
        assert_eq!(
            provider.extra_headers.get("X-Custom-Header"),
            Some(&"custom-value".to_string())
        );
    }

    #[test]
    fn test_with_proxy_preserves_timeout() {
        let provider = HttpProvider::new("test", "test-api-key", "https://api.example.com")
            .with_proxy("http://127.0.0.1:7890")
            .with_timeout(Duration::from_secs(60));

        assert_eq!(provider.proxy_url.as_deref(), Some("http://127.0.0.1:7890"));
        assert_eq!(provider.timeout, Duration::from_secs(60));
    }

    #[tokio::test]
    async fn post_json_sends_bearer_auth_and_decodes_body() {
        let (url, server) = serve_once(
            "HTTP/1.1 200 OK",
            &json!({ "data": [{ "b64_json": "aW1hZ2U=" }] }).to_string(),
        )
        .await;
        let provider = HttpProvider::new("openrouter", "test-api-key", url);

        let body = provider
            .post_json("/images", json!({ "model": "m", "prompt": "p" }))
            .await
            .expect("request should succeed");

        let raw = server.await.expect("test server should complete");
        let (headers, request_body) = raw
            .split_once("\r\n\r\n")
            .expect("request should contain headers and body");

        assert!(headers.starts_with("POST /images HTTP/1.1"));
        assert!(
            headers
                .to_ascii_lowercase()
                .contains("authorization: bearer test-api-key")
        );
        assert_eq!(
            serde_json::from_str::<Value>(request_body).unwrap(),
            json!({ "model": "m", "prompt": "p" })
        );
        assert_eq!(body["data"][0]["b64_json"], "aW1hZ2U=");
    }

    #[tokio::test]
    async fn post_json_maps_api_errors() {
        let (url, server) = serve_once(
            "HTTP/1.1 401 Unauthorized",
            &json!({ "error": { "code": 401, "message": "bad key" } }).to_string(),
        )
        .await;
        let provider = HttpProvider::new("test", "test-api-key", url);

        let error = provider
            .post_json("/chat/completions", json!({}))
            .await
            .expect_err("401 should fail");
        let _ = server.await;

        assert!(matches!(
            error,
            ProviderError::ApiError { code: 401, message } if message == "bad key"
        ));
    }

    #[tokio::test]
    async fn post_sse_yields_data_payloads_in_order() {
        let (url, server) = serve_once(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream",
            "data: {\"i\":1}\r\n\r\ndata: {\"i\":2}\r\n\r\ndata: [DONE]\r\n\r\n",
        )
        .await;
        let provider = HttpProvider::new("test", "test-api-key", url);

        let stream = provider
            .post_sse("/chat/completions", json!({ "stream": true }))
            .await
            .expect("stream request should succeed");
        let events = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .expect("stream should not fail");
        let _ = server.await;

        assert_eq!(events, vec!["{\"i\":1}", "{\"i\":2}", "[DONE]"]);
    }

    /// 起一个只服务一次请求的 HTTP 服务端，返回 (base_url, 请求原文句柄)。
    async fn serve_once(status_line: &str, response_body: &str) -> (String, tokio::task::JoinHandle<String>) {
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
                    && let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n")
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
