//! 传输层错误。
//!
//! 这里只描述「一次字节交换失败」的原因：连不上、响应解不开、服务端返回非 2xx。
//! 它不认识厂商、密钥方案和业务语义。

use std::time::Duration;

use thiserror::Error;

/// 传输层错误
#[derive(Debug, Error)]
pub enum TransportError {
    /// 底层 HTTP 客户端错误（当前实现基于 reqwest）
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    /// 请求构造失败（URL 或 header 非法）
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    /// 服务端返回非 2xx
    #[error("API error {status}: {message}")]
    Api {
        status: u16,
        message: String,
        retry_after: Option<Duration>,
    },
    /// 响应体不是合法 JSON
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    /// 响应流中途中断
    #[error("Stream error: {0}")]
    Stream(String),
    /// 缺少 API 密钥
    #[error("Missing API key")]
    MissingApiKey,
}
