use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use bytes::Bytes;
use serde_json::json;

use crate::endpoints::{OpenRouter, build_deepseek, build_openrouter};
use crate::providers::Provider;
use crate::transport::{HttpRequest, HttpResponse, Transport, TransportError};
use crate::{GenerateCapability, GenerateRequest, MediaSource, Message};

#[test]
fn deepseek_is_a_bare_openai_compatible_adapter() {
    let model = build_deepseek("dummy-key");

    assert_eq!(model.provider().name(), "deepseek");
    assert_eq!(model.provider().base_url(), "https://api.deepseek.com");
}

#[test]
fn openrouter_is_composed_from_one_provider() {
    let model = build_openrouter("dummy-key", None, None);

    assert_eq!(model.chat_adapter().provider().name(), "openrouter");
    assert_eq!(model.images_adapter().provider().name(), "openrouter");
    assert_eq!(
        model.images_adapter().provider().base_url(),
        model.chat_adapter().provider().base_url()
    );
}

#[test]
fn openrouter_config_headers_land_on_the_provider() {
    let model = build_openrouter(
        "dummy-key",
        Some("https://example.com".to_string()),
        Some("alaya".to_string()),
    );

    assert_eq!(
        model.chat_adapter().provider().header("HTTP-Referer"),
        Some("https://example.com")
    );
    assert_eq!(
        model.chat_adapter().provider().header("X-Title"),
        Some("alaya")
    );
}

#[derive(Default)]
struct RoutingTransport {
    paths: Mutex<Vec<String>>,
}

#[async_trait]
impl Transport for RoutingTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        self.paths.lock().unwrap().push(
            request
                .url
                .trim_start_matches("https://fake.test")
                .to_string(),
        );

        let body = if request.url.ends_with("/images") {
            json!({ "data": [{ "b64_json": "aW1hZ2U=", "media_type": "image/png" }] })
        } else {
            json!({
                "id": "resp_1",
                "object": "chat.completion",
                "created": 1,
                "model": "m",
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": "hello" },
                    "finish_reason": "stop"
                }]
            })
        };

        Ok(HttpResponse::buffered(200, Bytes::from(body.to_string())))
    }
}

#[tokio::test]
async fn openrouter_routes_each_modality_to_its_protocol() {
    let transport = Arc::new(RoutingTransport::default());
    let provider =
        Provider::new(transport.clone(), "openrouter", "https://fake.test").with_bearer("k");
    let model = OpenRouter::new(provider);

    let text_resp = model
        .generate(GenerateRequest::new("m", vec![Message::user("hi")]))
        .await
        .unwrap();
    let img_resp = model
        .generate(GenerateRequest::image(
            "krea/krea-2-medium-turbo",
            "landscape",
        ))
        .await
        .unwrap();

    assert_eq!(text_resp.message, Message::assistant("hello"));
    assert_eq!(
        img_resp.message.images(),
        vec![&MediaSource::base64("image/png", "aW1hZ2U=")]
    );
    assert_eq!(
        *transport.paths.lock().unwrap(),
        vec!["/chat/completions".to_string(), "/images".to_string()]
    );
}
