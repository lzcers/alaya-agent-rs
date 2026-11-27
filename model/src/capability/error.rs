use std::time::Duration;

use thiserror::Error;

use crate::transport::TransportError;

/// 模型调用错误。
///
/// 在能力边界把底层传输错误收敛成可编程的域语义（鉴权失败、限流、参数非法、
/// 不支持的模态/特性等），让上层 Agent 与 [`crate::router::ModelRouter`] 可以做重试与降级。
#[derive(Debug, Error)]
pub enum ModelError {
    /// 字节没能成功交换（连不上、超时、非 2xx、响应体不是 JSON）
    #[error("Transport error: {0}")]
    Transport(TransportError),
    /// 认证或授权失败（401 / 403）
    #[error("Authentication failed: {0}")]
    Auth(String),
    /// 触发上游限流（429）
    #[error("Rate limited: {message}")]
    RateLimited {
        message: String,
        retry_after: Option<Duration>,
    },
    /// 请求参数不合法
    #[error("Invalid request: {0}")]
    InvalidRequest(String),
    /// 超出上下文长度限制
    #[error("Context length exceeded: {0}")]
    ContextLengthExceeded(String),
    /// 触发安全策略拦截
    #[error("Content filtered: {0}")]
    ContentFiltered(String),
    /// 目标模型或协议不支持请求的模态或能力
    #[error("Unsupported feature or modality: {0}")]
    Unsupported(String),
    /// 字节拿到了，但没能映射成域类型（协议层的 wire → 域翻译失败）
    #[error("Decode error: {0}")]
    Decode(#[from] serde_json::Error),
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
            Self::RateLimited { .. } | Self::StreamError(_) => true,
            Self::Transport(TransportError::Api { status, .. }) => {
                *status == 408 || *status == 429 || (500..600).contains(status)
            }
            Self::Transport(TransportError::Http(_) | TransportError::Stream(_)) => true,
            _ => false,
        }
    }
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
            other => Self::Transport(other),
        }
    }
}

/// 把错误连同完整 cause 链格式化成一行，便于日志与上层展示。
pub fn format_model_error(error: &ModelError) -> String {
    let mut details = error.to_string();
    let mut source = std::error::Error::source(error);
    let mut index = 1;

    while let Some(error) = source {
        details.push_str(&format!("; caused by #{index}: {error}"));
        source = error.source();
        index += 1;
    }

    details.push_str(&format!("; debug: {error:?}"));
    details
}
