use thiserror::Error;

use crate::providers::ProviderError;

/// Model routing and execution errors.
#[derive(Debug, Error)]
pub enum RouterError {
    #[error("Provider error: {0}")]
    Provider(#[from] ProviderError),
    #[error("No response from model")]
    NoResponse,
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Model {model} does not support {capability}")]
    UnsupportedCapability {
        model: String,
        capability: &'static str,
    },
}

pub fn format_router_error(error: &RouterError) -> String {
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
    fn format_router_error_includes_error_chain_and_debug_details() {
        let serialization_error = serde_json::from_str::<serde_json::Value>("not-json")
            .expect_err("invalid JSON should fail");
        let error = RouterError::Provider(ProviderError::Serialization(serialization_error));

        let details = format_router_error(&error);

        assert!(details.contains("Provider error: Serialization error:"));
        assert!(details.contains("caused by #1: Serialization error:"));
        assert!(details.contains("caused by #2: expected ident"));
        assert!(details.contains("debug: Provider(Serialization("));
    }
}
