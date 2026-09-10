use thiserror::Error;

use crate::providers::ProviderError;

/// 模型调用错误。
///
/// 前三个变体由能力实现（协议层）产生；`ModelNotFound` 只在实现者是
/// [`crate::router::ModelRouter`] 时出现，表示该模型没有注册这个能力。
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),
    #[error("No response from model")]
    NoResponse,
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Model {model} is not registered for {capability}")]
    ModelNotFound {
        model: String,
        capability: &'static str,
    },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_model_error_includes_error_chain_and_debug_details() {
        let serialization_error = serde_json::from_str::<serde_json::Value>("not-json")
            .expect_err("invalid JSON should fail");
        let error = ModelError::Provider(ProviderError::Serialization(serialization_error));

        let details = format_model_error(&error);

        assert!(details.contains("Provider error: Serialization error:"));
        assert!(details.contains("caused by #1: Serialization error:"));
        assert!(details.contains("caused by #2: expected ident"));
        assert!(details.contains("debug: Provider(Serialization("));
    }
}
