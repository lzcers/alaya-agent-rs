use serde_json::Value;

/// 传输层错误。
///
/// 只描述「一次 HTTP 交互失败」的原因，不承载任何业务语义。
/// 能力层/协议层负责把它包装成面向调用方的错误。
#[derive(Debug)]
pub enum ProviderError {
    Request(reqwest::Error),
    Serialization(serde_json::Error),
    InvalidApiKey,
    ApiError { code: u16, message: String },
    MissingApiKey,
    StreamError(String),
}

impl std::error::Error for ProviderError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProviderError::Request(e) => Some(e),
            ProviderError::Serialization(e) => Some(e),
            _ => None,
        }
    }
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProviderError::Request(e) => write!(f, "Request error: {}", e),
            ProviderError::Serialization(e) => write!(f, "Serialization error: {}", e),
            ProviderError::InvalidApiKey => write!(f, "Invalid API key"),
            ProviderError::ApiError { code, message } => {
                write!(f, "API error {}: {}", code, message)
            }
            ProviderError::MissingApiKey => write!(f, "Missing API key"),
            ProviderError::StreamError(msg) => write!(f, "Stream error: {}", msg),
        }
    }
}

impl From<reqwest::Error> for ProviderError {
    fn from(e: reqwest::Error) -> Self {
        ProviderError::Request(e)
    }
}

impl From<serde_json::Error> for ProviderError {
    fn from(e: serde_json::Error) -> Self {
        ProviderError::Serialization(e)
    }
}

/// 解析 API 错误响应
///
/// 尝试从响应体中提取错误代码和消息
pub fn parse_api_error(body: &str, status: u16) -> ProviderError {
    if let Ok(error_json) = serde_json::from_str::<Value>(body) {
        let code = error_json["error"]["code"]
            .as_i64()
            .or_else(|| error_json["error"]["type"].as_str().map(|t| t.len() as i64))
            .unwrap_or(0) as u16;
        let message = error_json["error"]["message"]
            .as_str()
            .or_else(|| error_json["error"].as_str())
            .unwrap_or(body)
            .to_string();
        return ProviderError::ApiError { code, message };
    }
    ProviderError::ApiError {
        code: status,
        message: body.to_string(),
    }
}
