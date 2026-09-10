//! 与厂商无关的 HTTP 请求/响应类型。
//!
//! `HttpRequest` 是「字节 + 目标地址 + 请求头」的完整描述，`HttpResponse` 把响应体
//! 统一暴露成字节流——是否缓冲、怎么解释（JSON / SSE / 裸音频字节）由调用方决定。
//! 二进制响应因此天然可表达，不需要给传输层加新方法。

use bytes::Bytes;
use futures::stream::{BoxStream, StreamExt};
use reqwest::Method;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Serialize;
use serde::de::DeserializeOwned;

use super::TransportError;

/// 响应体字节流。
pub type BodyStream = BoxStream<'static, Result<Bytes, TransportError>>;

/// 一次 HTTP 请求。
#[derive(Debug)]
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: HeaderMap,
    pub body: Bytes,
}

impl HttpRequest {
    pub fn new(method: Method, url: impl Into<String>) -> Self {
        Self {
            method,
            url: url.into(),
            headers: HeaderMap::new(),
            body: Bytes::new(),
        }
    }

    pub fn post(url: impl Into<String>) -> Self {
        Self::new(Method::POST, url)
    }

    pub fn with_header(mut self, name: &str, value: impl AsRef<str>) -> Result<Self, TransportError> {
        let name = HeaderName::try_from(name)
            .map_err(|e| TransportError::InvalidRequest(format!("invalid header name: {e}")))?;
        let value = HeaderValue::from_str(value.as_ref())
            .map_err(|e| TransportError::InvalidRequest(format!("invalid header value: {e}")))?;
        self.headers.insert(name, value);
        Ok(self)
    }

    /// 用 JSON 作为请求体，并带上 `Content-Type`。
    pub fn with_json_body<T: Serialize + ?Sized>(mut self, body: &T) -> Result<Self, TransportError> {
        let encoded = serde_json::to_vec(body)?;
        self.headers
            .insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        self.body = Bytes::from(encoded);
        Ok(self)
    }

    /// 用原始字节作为请求体。二进制上传（音频、文件）走这里。
    pub fn with_bytes_body(
        mut self,
        content_type: &str,
        body: impl Into<Bytes>,
    ) -> Result<Self, TransportError> {
        let value = HeaderValue::from_str(content_type)
            .map_err(|e| TransportError::InvalidRequest(format!("invalid content type: {e}")))?;
        self.headers.insert(CONTENT_TYPE, value);
        self.body = body.into();
        Ok(self)
    }

    /// 声明期望事件流响应（SSE）。
    pub fn accepting_sse(mut self) -> Result<Self, TransportError> {
        self.headers.insert(
            reqwest::header::ACCEPT,
            HeaderValue::from_static("text/event-stream"),
        );
        Ok(self)
    }
}

/// 一次 HTTP 响应。
///
/// 不区分「缓冲」和「流式」：响应体永远是一个字节流，`bytes()` / `json()` 负责收集，
/// `into_body()` 把流原样交出去。
pub struct HttpResponse {
    status: u16,
    headers: HeaderMap,
    body: BodyStream,
}

impl HttpResponse {
    pub fn new(status: u16, headers: HeaderMap, body: BodyStream) -> Self {
        Self {
            status,
            headers,
            body,
        }
    }

    /// 用一段已知字节构造响应（测试替身与内部复用）。
    pub fn buffered(status: u16, body: impl Into<Bytes>) -> Self {
        let body = body.into();
        Self {
            status,
            headers: HeaderMap::new(),
            body: Box::pin(futures::stream::once(async move { Ok(body) })),
        }
    }

    pub fn status(&self) -> u16 {
        self.status
    }

    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    pub fn headers(&self) -> &HeaderMap {
        &self.headers
    }

    /// 交出响应体流，不做任何解释。
    pub fn into_body(self) -> BodyStream {
        self.body
    }

    /// 收齐整个响应体。
    pub async fn bytes(self) -> Result<Bytes, TransportError> {
        let mut collected = Vec::new();
        let mut body = self.body;
        while let Some(chunk) = body.next().await {
            collected.extend_from_slice(&chunk?);
        }
        Ok(Bytes::from(collected))
    }

    /// 收齐响应体并按 UTF-8 解码。
    pub async fn text(self) -> Result<String, TransportError> {
        Ok(String::from_utf8_lossy(&self.bytes().await?).into_owned())
    }

    /// 收齐响应体并反序列化。
    pub async fn json<T: DeserializeOwned>(self) -> Result<T, TransportError> {
        let bytes = self.bytes().await?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn json_body_sets_content_type_and_encodes() {
        let request = HttpRequest::post("https://example.com/x")
            .with_json_body(&json!({ "a": 1 }))
            .unwrap();

        assert_eq!(request.headers.get(CONTENT_TYPE).unwrap(), "application/json");
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
            json!({ "a": 1 })
        );
    }

    #[tokio::test]
    async fn buffered_response_collects_once() {
        let response = HttpResponse::buffered(200, Bytes::from_static(b"{\"ok\":true}"));

        assert!(response.is_success());
        assert_eq!(response.json::<serde_json::Value>().await.unwrap()["ok"], true);
    }

    #[tokio::test]
    async fn binary_body_is_expressible() {
        let request = HttpRequest::post("https://example.com/speech")
            .with_bytes_body("audio/mpeg", Bytes::from_static(b"\xff\xfb\x90"))
            .unwrap();

        assert_eq!(request.body.len(), 3);
        assert_eq!(
            request.headers.get(CONTENT_TYPE).unwrap(),
            "audio/mpeg"
        );
    }
}
