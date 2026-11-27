use std::sync::{Arc, Mutex};

use alaya_model::providers::Provider;
use alaya_model::transport::{HttpRequest, HttpResponse, Transport, TransportError};
use bytes::Bytes;
use futures::StreamExt;
use reqwest::Method;
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE, HeaderMap};
use serde_json::{Value, json};

#[derive(Debug, Clone)]
struct Seen {
    method: Method,
    url: String,
    headers: HeaderMap,
    body: Bytes,
}

#[derive(Default)]
struct ScriptedTransport {
    seen: Mutex<Vec<Seen>>,
    status: u16,
    body: String,
}

impl ScriptedTransport {
    fn responding(status: u16, body: Value) -> Arc<Self> {
        Arc::new(Self {
            status,
            body: body.to_string(),
            ..Default::default()
        })
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl Transport for ScriptedTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        self.seen.lock().unwrap().push(Seen {
            method: request.method.clone(),
            url: request.url.clone(),
            headers: request.headers.clone(),
            body: request.body.clone(),
        });
        Ok(HttpResponse::buffered(self.status, self.body.clone()))
    }
}

fn provider(transport: Arc<ScriptedTransport>) -> Provider {
    Provider::new(transport, "acme", "https://api.acme.test/v1/").with_bearer("secret")
}

#[tokio::test]
async fn post_json_joins_url_injects_headers_and_decodes() {
    let transport = ScriptedTransport::responding(200, json!({ "ok": true }));
    let provider = provider(transport.clone()).with_header("x-acme-tenant", "t1");

    let value: Value = provider
        .post_json("/generate", &json!({ "prompt": "hi" }))
        .await
        .unwrap();

    assert_eq!(value["ok"], true);
    let seen = transport.seen();
    assert_eq!(seen[0].method, Method::POST);
    assert_eq!(seen[0].url, "https://api.acme.test/v1/generate");
    assert_eq!(seen[0].headers.get(AUTHORIZATION).unwrap(), "Bearer secret");
    assert_eq!(seen[0].headers.get("x-acme-tenant").unwrap(), "t1");
    assert_eq!(
        seen[0].headers.get(CONTENT_TYPE).unwrap(),
        "application/json"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&seen[0].body).unwrap(),
        json!({ "prompt": "hi" })
    );
}

#[tokio::test]
async fn non_success_becomes_api_error_with_extracted_message() {
    let transport = ScriptedTransport::responding(
        401,
        json!({ "error": { "code": 401, "message": "bad key" } }),
    );
    let provider = provider(transport);

    let error = provider
        .post_json::<_, Value>("/generate", &json!({}))
        .await
        .expect_err("401 should fail");

    assert!(matches!(
        error,
        TransportError::Api { status: 401, message, .. } if message == "bad key"
    ));
}

#[tokio::test]
async fn error_message_falls_back_to_raw_body() {
    let transport = Arc::new(ScriptedTransport {
        status: 500,
        body: "upstream exploded".to_string(),
        ..Default::default()
    });

    let error = provider(transport)
        .post_json::<_, Value>("/x", &json!({}))
        .await
        .expect_err("500 should fail");

    assert!(matches!(
        error,
        TransportError::Api { status: 500, message, .. } if message == "upstream exploded"
    ));
}

#[tokio::test]
async fn a_provider_without_bearer_sends_no_authorization_header() {
    let transport = ScriptedTransport::responding(200, json!({}));
    let provider = Provider::new(
        transport.clone(),
        "anthropic",
        "https://api.anthropic.com/v1",
    )
    .with_header_auth("x-api-key", "secret");

    let _: Value = provider.post_json("/messages", &json!({})).await.unwrap();

    let seen = transport.seen();
    assert!(seen[0].headers.get(AUTHORIZATION).is_none());
    assert_eq!(seen[0].headers.get("x-api-key").unwrap(), "secret");
}

#[tokio::test]
async fn post_sse_frames_events_from_the_response_body() {
    let transport = Arc::new(ScriptedTransport {
        status: 200,
        body: "event: msg\ndata: {\"i\":1}\n\ndata: [DONE]\n\n".to_string(),
        ..Default::default()
    });
    let provider = provider(transport.clone());

    let events = provider
        .post_sse("/stream", &json!({ "stream": true }))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(events.len(), 2);
    assert_eq!(events[0].event.as_deref(), Some("msg"));
    assert_eq!(events[0].data, "{\"i\":1}");
    assert_eq!(events[1].data, "[DONE]");
    assert_eq!(
        transport.seen()[0].headers.get(ACCEPT).unwrap(),
        "text/event-stream"
    );
}
