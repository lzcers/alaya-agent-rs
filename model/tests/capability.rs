use alaya_model::{
    GenerateRequest, ImageOptions, Modality, ModelError, OutputSpec, ReasoningConfig,
    ReasoningEffort, TextFormat, ToolDef,
};
use serde_json::json;

#[test]
fn request_builders_configure_domain_fields() {
    let request = GenerateRequest::new("model", Vec::new())
        .with_outputs([
            OutputSpec::json_object(),
            OutputSpec::image(ImageOptions::new().with_resolution("1K")),
        ])
        .with_reasoning(ReasoningConfig::Effort {
            level: ReasoningEffort::High,
        })
        .with_temperature(0.2)
        .with_max_tokens(512);

    assert_eq!(
        request.output_modalities(),
        vec![Modality::Text, Modality::Image]
    );
    assert_eq!(request.config.text_format(), Some(&TextFormat::JsonObject));
    assert_eq!(
        request
            .config
            .image_options()
            .and_then(|o| o.resolution.as_deref()),
        Some("1K")
    );
    assert_eq!(request.config.temperature, Some(0.2));
    assert_eq!(request.config.max_tokens, Some(512));
    assert_eq!(
        request.config.reasoning,
        ReasoningConfig::Effort {
            level: ReasoningEffort::High
        }
    );
}

#[test]
fn tools_stay_domain_typed() {
    let tool = ToolDef {
        name: "read_file".to_string(),
        description: "read".to_string(),
        parameters: json!({ "type": "object" }),
    };
    let request = GenerateRequest::new("model", Vec::new()).with_tools(Some(vec![tool]));

    assert_eq!(request.tools.unwrap()[0].name, "read_file");
}

#[test]
fn model_error_detailed_includes_display_and_debug() {
    let error = ModelError::Decode("invalid json".to_string());
    let details = error.detailed();

    assert!(details.starts_with("Decode error: invalid json"));
    assert!(details.contains("debug: Decode("));
}

#[test]
fn retryable_error_classification_covers_transient_failures() {
    assert!(ModelError::Network("timeout".into()).is_retryable());
    assert!(
        ModelError::Api {
            status: 503,
            message: "unavailable".into()
        }
        .is_retryable()
    );
    assert!(
        ModelError::RateLimited {
            message: "slow down".into(),
            retry_after: None
        }
        .is_retryable()
    );
    assert!(!ModelError::Auth("bad key".into()).is_retryable());
    assert!(!ModelError::MissingApiKey.is_retryable());
}
