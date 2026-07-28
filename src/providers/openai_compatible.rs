use async_trait::async_trait;
use futures::{StreamExt, stream::BoxStream};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap};
use std::collections::HashMap;
use std::time::Duration;
use tracing::info;

use super::{
    ImageGenerationRequest, ImageGenerationResponse, Provider, ProviderError, Request, Response,
    StreamResponse, parse_api_error,
};

/// OpenAI 兼容的 Provider 实现
///
/// 这是一个通用的 HTTP provider，可以用于任何兼容 OpenAI API 格式的服务。
/// 包括 DeepSeek、OpenRouter、Groq 等提供商。
pub struct OpenAICompatibleProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    extra_headers: HashMap<String, String>,
    name: String,
    timeout: Duration,
    proxy_url: Option<String>,
}

impl OpenAICompatibleProvider {
    /// 创建新的 OpenAI 兼容 Provider
    ///
    /// # Arguments
    /// * `name` - Provider 名称（用于日志和调试）
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

    /// 构建请求头
    fn build_headers(&self) -> HeaderMap {
        use reqwest::header::HeaderName;

        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", self.api_key)
                .parse()
                .expect("Invalid API key format"),
        );
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());

        // 添加额外的请求头
        for (key, value) in &self.extra_headers {
            if let Ok(name) = HeaderName::try_from(key.as_str())
                && let Ok(val) = value.parse()
            {
                headers.insert(name, val);
            }
        }

        headers
    }
}

fn build_http_client(timeout: Duration, proxy_url: Option<&str>) -> reqwest::Client {
    let mut builder = reqwest::Client::builder().timeout(timeout);

    if let Some(proxy_url) = proxy_url.map(str::trim).filter(|url| !url.is_empty()) {
        builder = builder.proxy(reqwest::Proxy::all(proxy_url).expect("Invalid proxy URL"));
    }

    builder.build().expect("Failed to build HTTP client")
}

fn normalize_crlf(buffer: &mut Vec<u8>) {
    let mut read = 0;
    let mut write = 0;

    while read < buffer.len() {
        if buffer[read] == b'\r' && buffer.get(read + 1) == Some(&b'\n') {
            buffer[write] = b'\n';
            read += 2;
        } else {
            buffer[write] = buffer[read];
            read += 1;
        }
        write += 1;
    }

    buffer.truncate(write);
}

fn drain_sse_frames(buffer: &mut Vec<u8>) -> Vec<Vec<u8>> {
    normalize_crlf(buffer);

    let mut frames = Vec::new();
    while let Some(idx) = buffer.windows(2).position(|window| window == b"\n\n") {
        let remaining = buffer.split_off(idx + 2);
        buffer.truncate(idx);
        let frame = std::mem::replace(buffer, remaining);
        if frame.iter().any(|byte| !byte.is_ascii_whitespace()) {
            frames.push(frame);
        }
    }
    frames
}

fn parse_sse_frame(frame: &[u8]) -> Option<StreamResponse> {
    let frame = std::str::from_utf8(frame).ok()?;
    let payload = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
        .collect::<Vec<_>>()
        .join("\n");

    if payload.is_empty() || payload == "[DONE]" {
        return None;
    }

    serde_json::from_str::<StreamResponse>(&payload).ok()
}

#[async_trait]
impl Provider for OpenAICompatibleProvider {
    /// 发送非流式请求
    async fn chat(&self, request: Request) -> Result<Response, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);
        let headers = self.build_headers();
        let proxy_url = self.proxy_url.as_deref().unwrap_or("<none>");
        let modalities = request
            .extra
            .get("modalities")
            .map_or_else(|| "<none>".to_string(), ToString::to_string);
        info!(
            provider = %self.name,
            model = %request.model,
            stream = false,
            modalities = %modalities,
            url = %url,
            proxy_configured = self.proxy_url.is_some(),
            proxy_url = %proxy_url,
            "sending provider chat request"
        );

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(parse_api_error(&body, status.as_u16()));
        }

        let response: Response = serde_json::from_str(&body)?;
        Ok(response)
    }

    /// 发送流式请求
    async fn chat_stream(
        &self,
        request: Request,
    ) -> Result<BoxStream<'static, StreamResponse>, ProviderError> {
        let url = format!("{}/chat/completions", self.base_url);
        let headers = self.build_headers();
        let proxy_url = self.proxy_url.as_deref().unwrap_or("<none>");
        let modalities = request
            .extra
            .get("modalities")
            .map_or_else(|| "<none>".to_string(), ToString::to_string);
        info!(
            provider = %self.name,
            model = %request.model,
            stream = true,
            modalities = %modalities,
            url = %url,
            proxy_configured = self.proxy_url.is_some(),
            proxy_url = %proxy_url,
            "sending provider chat request"
        );

        let mut stream_request = request;
        stream_request.stream = Some(true);

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&stream_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(parse_api_error(&body, status.as_u16()));
        }

        let stream = response
            .bytes_stream()
            .scan(Vec::new(), |buffer, chunk_result| {
                let chunk = match chunk_result {
                    Ok(c) => c,
                    Err(_) => return futures::future::ready(Some(vec![])),
                };

                buffer.extend_from_slice(&chunk);

                let parsed = drain_sse_frames(buffer)
                    .into_iter()
                    .filter_map(|frame| parse_sse_frame(&frame))
                    .collect::<Vec<_>>();

                futures::future::ready(Some(parsed))
            })
            .flat_map(futures::stream::iter);

        Ok(Box::pin(stream))
    }

    async fn generate_image(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse, ProviderError> {
        let url = format!("{}/images", self.base_url);
        let headers = self.build_headers();
        info!(
            provider = %self.name,
            model = %request.model,
            resolution = request.resolution.as_deref().unwrap_or("<none>"),
            aspect_ratio = request.aspect_ratio.as_deref().unwrap_or("<none>"),
            url = %url,
            proxy_configured = self.proxy_url.is_some(),
            "sending provider image request"
        );

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(parse_api_error(&body, status.as_u16()));
        }

        Ok(serde_json::from_str(&body)?)
    }

    /// Provider 名称
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
        let provider =
            OpenAICompatibleProvider::new("test", "test-api-key", "https://api.example.com");

        assert_eq!(provider.name(), "test");
        assert_eq!(provider.api_key, "test-api-key");
        assert_eq!(provider.base_url, "https://api.example.com");
    }

    #[test]
    fn test_with_extra_headers() {
        let mut extra = HashMap::new();
        extra.insert("X-Custom-Header".to_string(), "custom-value".to_string());

        let provider =
            OpenAICompatibleProvider::new("test", "test-api-key", "https://api.example.com")
                .with_extra_headers(extra);

        assert_eq!(provider.extra_headers.len(), 1);
        assert_eq!(
            provider.extra_headers.get("X-Custom-Header"),
            Some(&"custom-value".to_string())
        );
    }

    #[test]
    fn test_with_timeout() {
        let provider =
            OpenAICompatibleProvider::new("test", "test-api-key", "https://api.example.com")
                .with_timeout(Duration::from_secs(60));

        // 验证 provider 创建成功
        assert_eq!(provider.name(), "test");
    }

    #[test]
    fn test_with_proxy_preserves_timeout() {
        let provider =
            OpenAICompatibleProvider::new("test", "test-api-key", "https://api.example.com")
                .with_proxy("http://127.0.0.1:7890")
                .with_timeout(Duration::from_secs(60));

        assert_eq!(provider.proxy_url.as_deref(), Some("http://127.0.0.1:7890"));
        assert_eq!(provider.timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_drain_sse_frames_handles_split_chunks() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"data: {\"id\":\"abc\"");
        assert!(drain_sse_frames(&mut buffer).is_empty());

        buffer.extend_from_slice(b",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"system_fingerprint\":null,\"choices\":[],\"usage\":null}\n\n");
        let frames = drain_sse_frames(&mut buffer);

        assert_eq!(frames.len(), 1);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_sse_frame_preserves_multibyte_content_across_every_chunk_boundary() {
        let frame = format!(
            "data: {}\r\n\r\n",
            json!({
                "id": "abc",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "system_fingerprint": null,
                "choices": [{
                    "index": 0,
                    "delta": { "content": "铜片🙂" },
                    "finish_reason": null
                }],
                "usage": null
            })
        );

        for split_at in 0..=frame.len() {
            let mut buffer = Vec::new();
            buffer.extend_from_slice(&frame.as_bytes()[..split_at]);
            let mut frames = drain_sse_frames(&mut buffer);

            buffer.extend_from_slice(&frame.as_bytes()[split_at..]);
            frames.extend(drain_sse_frames(&mut buffer));

            assert_eq!(frames.len(), 1, "split_at={split_at}");
            let parsed = parse_sse_frame(&frames[0]).expect("frame should parse");
            assert_eq!(
                parsed.choices[0].delta.content.as_deref(),
                Some("铜片🙂"),
                "split_at={split_at}"
            );
            assert!(buffer.is_empty(), "split_at={split_at}");
        }
    }

    #[test]
    fn test_parse_sse_frame_ignores_done_and_parses_json() {
        assert!(parse_sse_frame(b"data: [DONE]").is_none());

        let frame = format!(
            "data: {}\n",
            json!({
                "id": "abc",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "test-model",
                "system_fingerprint": null,
                "choices": [{
                    "index": 0,
                    "delta": { "content": "hi" },
                    "finish_reason": "stop"
                }],
                "usage": null
            })
        );

        let parsed = parse_sse_frame(frame.as_bytes()).expect("frame should parse");
        assert_eq!(parsed.choices.len(), 1);
        assert_eq!(parsed.choices[0].delta.content.as_deref(), Some("hi"));
        assert_eq!(parsed.choices[0].finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn image_generation_uses_dedicated_images_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener should bind");
        let address = listener
            .local_addr()
            .expect("test listener should have an address");
        let response_body = json!({
            "data": [{
                "b64_json": "aW1hZ2U=",
                "media_type": "image/png"
            }]
        })
        .to_string();
        let server = tokio::spawn(capture_one_request(listener, response_body));
        let provider = OpenAICompatibleProvider::new(
            "openrouter",
            "test-api-key",
            format!("http://{address}/api/v1"),
        );

        let response = provider
            .generate_image(
                ImageGenerationRequest::new("krea/krea-2-medium-turbo", "a cinematic landscape")
                    .with_resolution("1K")
                    .with_aspect_ratio("16:9"),
            )
            .await
            .expect("image request should succeed");
        let raw_request = server.await.expect("test server should complete");
        let (headers, body) = raw_request
            .split_once("\r\n\r\n")
            .expect("request should contain headers and body");
        let body: serde_json::Value =
            serde_json::from_str(body).expect("request body should be JSON");

        assert!(headers.starts_with("POST /api/v1/images HTTP/1.1"));
        assert_eq!(body["model"], "krea/krea-2-medium-turbo");
        assert_eq!(body["prompt"], "a cinematic landscape");
        assert_eq!(body["resolution"], "1K");
        assert_eq!(body["aspect_ratio"], "16:9");
        assert!(body.get("messages").is_none());
        assert!(body.get("modalities").is_none());
        assert!(body.get("image_config").is_none());
        assert_eq!(response.data.len(), 1);
        assert_eq!(response.data[0].b64_json.as_deref(), Some("aW1hZ2U="));
    }

    async fn capture_one_request(listener: TcpListener, response_body: String) -> String {
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

        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        socket
            .write_all(response.as_bytes())
            .await
            .expect("test server should write response");

        String::from_utf8(request).expect("request should be UTF-8")
    }
}
