use std::time::Duration;

use thiserror::Error;

use crate::transport::TransportError;

/// 模型调用错误。
///
/// 对外只暴露可编程的域语义（配置缺失、鉴权失败、限流、上游 API 错误、
/// 网络异常、参数非法、不支持的模态等），不泄漏底层传输库或协议实现细节。
#[derive(Debug, Error)]
pub enum ModelError {
    /// 缺少必需配置（如环境变量中的 API Key）
    #[error("Missing API key")]
    MissingApiKey,
    /// 认证或授权失败（401 / 403）
    #[error("Authentication failed: {0}")]
    Auth(String),
    /// 触发上游限流（429）
    #[error("Rate limited: {message}")]
    RateLimited {
        message: String,
        retry_after: Option<Duration>,
    },
    /// 上游服务端返回非 2xx 错误（401 / 403 / 429 除外）
    #[error("API error {status}: {message}")]
    Api { status: u16, message: String },
    /// 网络连接、超时或底层字节流中断
    #[error("Network error: {0}")]
    Network(String),
    /// 请求参数不合法
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    /// 目标模型或协议不支持请求的模态或能力
    #[error("Unsupported feature or modality: {0}")]
    Unsupported(String),
    /// 响应解析失败
    #[error("Decode error: {0}")]
    Decode(String),
    #[error("No response from model")]
    NoResponse,
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Model `{model}` is not registered")]
    ModelNotFound { model: String },
}

impl ModelError {
    /// 判断错误是否属于可重试 / 可切换到备选厂商的临时性故障。
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::RateLimited { .. } | Self::Network(_) | Self::StreamError(_) => true,
            Self::Api { status, .. } => {
                *status == 408 || *status == 429 || (500..600).contains(status)
            }
            _ => false,
        }
    }

    /// 格式化详细错误信息（含 cause 链与调试信息），便于日志与上层展示。
    pub fn detailed(&self) -> String {
        format!("{}; debug: {self:?}", format_error_chain(self))
    }
}

fn format_error_chain(error: &dyn std::error::Error) -> String {
    let mut details = error.to_string();
    let mut source = error.source();
    let mut index = 1;

    while let Some(err) = source {
        details.push_str(&format!("; caused by #{index}: {err}"));
        source = err.source();
        index += 1;
    }

    details
}

impl From<TransportError> for ModelError {
    fn from(error: TransportError) -> Self {
        match error {
            TransportError::Api {
                status: 401 | 403,
                message,
                ..
            } => Self::Auth(message),
            TransportError::Api {
                status: 429,
                message,
                retry_after,
            } => Self::RateLimited {
                message,
                retry_after,
            },
            TransportError::Api {
                status, message, ..
            } => Self::Api { status, message },
            TransportError::InvalidRequest(message) => Self::InvalidRequest(message),
            TransportError::Serialization(err) => Self::Decode(format_error_chain(&err)),
            TransportError::Http(err) => Self::Network(format_error_chain(&err)),
            TransportError::Stream(message) => Self::Network(message),
        }
    }
}
