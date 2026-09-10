//! 传输层错误。
//!
//! 这里只描述「一次字节交换失败」的原因：连不上、响应解不开、服务端返回非 2xx。
//! 它不认识厂商、密钥方案和业务语义。

/// 传输层错误
#[derive(Debug)]
pub enum TransportError {
    /// 底层 HTTP 客户端错误（当前实现基于 reqwest）
    Http(reqwest::Error),
    /// 请求构造失败（URL 或 header 非法）
    InvalidRequest(String),
    /// 服务端返回非 2xx
    Api { status: u16, message: String },
    /// 响应体不是合法 JSON
    Serialization(serde_json::Error),
    /// 响应流中途中断
    Stream(String),
    /// 缺少 API 密钥
    MissingApiKey,
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TransportError::Http(e) => Some(e),
            TransportError::Serialization(e) => Some(e),
            _ => None,
        }
    }
}

impl std::fmt::Display for TransportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransportError::Http(e) => write!(f, "HTTP error: {}", e),
            TransportError::InvalidRequest(message) => write!(f, "Invalid request: {}", message),
            TransportError::Api { status, message } => {
                write!(f, "API error {}: {}", status, message)
            }
            TransportError::Serialization(e) => write!(f, "Serialization error: {}", e),
            TransportError::Stream(message) => write!(f, "Stream error: {}", message),
            TransportError::MissingApiKey => write!(f, "Missing API key"),
        }
    }
}

impl From<reqwest::Error> for TransportError {
    fn from(e: reqwest::Error) -> Self {
        TransportError::Http(e)
    }
}

impl From<serde_json::Error> for TransportError {
    fn from(e: serde_json::Error) -> Self {
        TransportError::Serialization(e)
    }
}
