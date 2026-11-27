use alaya_model::capability::{
    GenerateRequest, ImageOptions, ModelError, OutputSpec, ReasoningConfig, ReasoningEffort,
    TextFormat, format_model_error,
};
use alaya_model::transport::TransportError;
use alaya_model::{Modality, ToolDef};
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
fn format_model_error_includes_error_chain_and_debug_details() {
    let serialization_error = serde_json::from_str::<serde_json::Value>("not-json")
        .expect_err("invalid JSON should fail");
    let error = ModelError::from(TransportError::Serialization(serialization_error));

    let details = format_model_error(&error);

    assert!(details.contains("Transport error: Serialization error:"));
    assert!(details.contains("debug: Transport(Serialization("));
}

#[test]
fn serde_failures_map_to_decode_not_transport() {
    let error: ModelError = serde_json::from_str::<serde_json::Value>("not-json")
        .expect_err("invalid JSON should fail")
        .into();

    assert!(matches!(error, ModelError::Decode(_)));
    assert!(format_model_error(&error).starts_with("Decode error:"));
}
