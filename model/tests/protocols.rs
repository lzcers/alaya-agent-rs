use std::sync::{Arc, Mutex};

use alaya_model::capability::{
    FinishReason, GenerateCapability, GenerateRequest, ImageOptions, ModelError, OutputSpec,
    PartDelta, ReasoningConfig, ReasoningEffort, StreamEvent, collect_stream,
};
use alaya_model::protocols::openai::types::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, ChoiceMessage, ChunkDelta,
    ToolCall as OpenAiToolCall, ToolCallFunction,
};
use alaya_model::protocols::openrouter::generated_image_to_source;
use alaya_model::protocols::openrouter::types::GeneratedImage;
use alaya_model::protocols::{OpenAiCompatible, OpenRouterImages};
use alaya_model::providers::Provider;
use alaya_model::transport::{HttpRequest, HttpResponse, Transport, TransportError};
use alaya_model::{ContentPart, MediaSource, Message, ToolCall, ToolDef};
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use serde_json::{Value, json};

const BASE_URL: &str = "https://fake.test/v1";

#[derive(Default)]
struct FakeTransport {
    requests: Mutex<Vec<(String, Value)>>,
    json: Option<Value>,
    events: Vec<String>,
}

impl FakeTransport {
    fn with_json(json: Value) -> Arc<Self> {
        Arc::new(Self {
            json: Some(json),
            ..Default::default()
        })
    }

    fn with_events(events: Vec<Value>) -> Arc<Self> {
        Self::with_raw_events(events.into_iter().map(|value| value.to_string()).collect())
    }

    fn with_raw_events(events: Vec<String>) -> Arc<Self> {
        Arc::new(Self {
            events,
            ..Default::default()
        })
    }

    fn recorded(&self) -> Vec<(String, Value)> {
        self.requests.lock().unwrap().clone()
    }

    fn recorded_paths(&self) -> Vec<String> {
        self.recorded()
            .into_iter()
            .map(|(url, _)| url.trim_start_matches(BASE_URL).to_string())
            .collect()
    }

    fn first_body(&self) -> Value {
        self.recorded()
            .into_iter()
            .next()
            .map(|(_, body)| body)
            .unwrap_or(Value::Null)
    }
}

fn fake_provider(transport: Arc<FakeTransport>) -> Provider {
    Provider::new(transport, "fake", BASE_URL).with_bearer("test-key")
}

#[async_trait]
impl Transport for FakeTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        let body = if request.body.is_empty() {
            Value::Null
        } else {
            serde_json::from_slice(&request.body).unwrap_or(Value::Null)
        };
        self.requests
            .lock()
            .unwrap()
            .push((request.url.clone(), body));

        if let Some(json) = &self.json {
            return Ok(HttpResponse::buffered(200, Bytes::from(json.to_string())));
        }

        let mut body = String::new();
        for event in &self.events {
            body.push_str("data: ");
            body.push_str(event);
            body.push_str("\n\n");
        }
        Ok(HttpResponse::buffered(200, Bytes::from(body)))
    }
}

// ============================================================================
// OpenAI protocol adapter
// ============================================================================

fn chat_wire_response(content: &str) -> Value {
    json!({
        "id": "resp_1",
        "object": "chat.completion",
        "created": 1,
        "model": "m",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }]
    })
}

fn audio_chunk(data: &str, transcript: &str) -> Value {
    json!({
        "id": "chunk",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "google/lyria-3-clip-preview",
        "system_fingerprint": null,
        "choices": [{
            "index": 0,
            "delta": { "audio": { "data": data, "transcript": transcript } },
            "finish_reason": null
        }]
    })
}

#[tokio::test]
async fn generate_posts_to_chat_completions_and_maps_to_domain_response() {
    let transport = FakeTransport::with_json(chat_wire_response("hello"));
    let model = OpenAiCompatible::new(fake_provider(transport.clone()));

    let response = model
        .generate(GenerateRequest::new("m", vec![Message::user("hi")]))
        .await
        .unwrap();

    assert_eq!(response.message, Message::assistant("hello"));
    assert_eq!(response.finish_reason, Some(FinishReason::Stop));
    assert_eq!(transport.recorded_paths(), vec!["/chat/completions"]);
    assert_eq!(transport.first_body()["model"], "m");
    assert!(transport.first_body().get("stream").is_none());
}

#[tokio::test]
async fn generate_stream_decodes_events_and_skips_done_marker() {
    let transport = FakeTransport::with_raw_events(vec![
        json!({
            "id": "1", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "system_fingerprint": null,
            "choices": [{ "index": 0, "delta": { "content": "he" }, "finish_reason": null }]
        })
        .to_string(),
        "[DONE]".to_string(),
        json!({
            "id": "2", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "system_fingerprint": null,
            "choices": [{ "index": 0, "delta": { "content": "llo" }, "finish_reason": "stop" }]
        })
        .to_string(),
    ]);
    let model = OpenAiCompatible::new(fake_provider(transport));

    let events: Vec<StreamEvent> = model
        .generate_stream(GenerateRequest::new("m", vec![Message::user("hi")]))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(
        events,
        vec![
            StreamEvent::Delta(PartDelta::Text {
                text: "he".to_string()
            }),
            StreamEvent::Delta(PartDelta::Text {
                text: "llo".to_string()
            }),
            StreamEvent::Finished {
                reason: Some(FinishReason::Stop)
            },
        ]
    );
}

#[tokio::test]
async fn stream_request_forces_stream_flag_at_the_protocol_layer() {
    let transport = FakeTransport::with_events(vec![]);
    let model = OpenAiCompatible::new(fake_provider(transport.clone()));

    model
        .generate_stream(GenerateRequest::new("m", vec![]))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;

    assert_eq!(transport.first_body()["stream"], true);
}

#[tokio::test]
async fn audio_generate_builds_modalities_and_collects_streamed_chunks() {
    let transport = FakeTransport::with_events(vec![
        audio_chunk("abc", "hello "),
        audio_chunk("def", "world"),
    ]);
    let model = OpenAiCompatible::new(fake_provider(transport.clone()));

    let response = model
        .generate(GenerateRequest::audio(
            "google/lyria-3-clip-preview",
            "Generate a short piano loop",
        ))
        .await
        .unwrap();

    let audios = response.message.audios();
    assert_eq!(audios.len(), 1);
    assert_eq!(audios[0].0, &MediaSource::base64("audio/wav", "abcdef"));
    assert_eq!(audios[0].1, Some("hello world"));

    let body = transport.first_body();
    assert_eq!(transport.recorded_paths(), vec!["/chat/completions"]);
    assert_eq!(body["stream"], true);
    assert_eq!(body["modalities"], json!(["text", "audio"]));
    assert_eq!(body["audio"], json!({ "format": "wav" }));
    assert_eq!(
        body["messages"][0]["content"],
        "Generate a short piano loop"
    );
}

#[tokio::test]
async fn streaming_tool_call_fragments_become_one_complete_domain_call() {
    let chunk = |delta: Value, finish: Value| {
        json!({
            "id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
            "system_fingerprint": null,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }]
        })
        .to_string()
    };
    let transport = FakeTransport::with_raw_events(vec![
        chunk(
            json!({ "tool_calls": [{ "index": 0, "id": "call_1", "type": "function",
                    "function": { "name": "read_file", "arguments": "" } }] }),
            Value::Null,
        ),
        chunk(
            json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "{\"path\":" } }] }),
            Value::Null,
        ),
        chunk(
            json!({ "tool_calls": [{ "index": 0, "function": { "arguments": "\"a.txt\"}" } }] }),
            Value::Null,
        ),
        chunk(json!({}), json!("tool_calls")),
    ]);
    let model = OpenAiCompatible::new(fake_provider(transport));

    let response = collect_stream(
        model
            .generate_stream(GenerateRequest::new("m", vec![Message::user("hi")]))
            .await
            .unwrap(),
    )
    .await
    .unwrap();

    assert_eq!(response.finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(
        response.message.tool_calls(),
        vec![&ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({ "path": "a.txt" }),
        }]
    );
}

#[tokio::test]
async fn assistant_tool_calls_round_trip_back_into_wire_messages() {
    let transport = FakeTransport::with_json(chat_wire_response("ok"));
    let model = OpenAiCompatible::new(fake_provider(transport.clone()));

    let messages = vec![
        Message::user("hi"),
        Message::assistant_parts(vec![ContentPart::ToolCall(ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({ "path": "a.txt" }),
        })]),
        Message::tool_result("call_1", "{}"),
    ];

    model
        .generate(GenerateRequest::new("m", messages))
        .await
        .unwrap();

    let body = transport.first_body();
    let call = &body["messages"][1]["tool_calls"][0];
    assert_eq!(call["id"], "call_1");
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "read_file");
    assert_eq!(call["function"]["arguments"], "{\"path\":\"a.txt\"}");
    assert!(call.get("name").is_none(), "不应泄漏域侧平铺字段");
    assert_eq!(body["messages"][2]["role"], "tool");
    assert_eq!(body["messages"][2]["tool_call_id"], "call_1");
}

#[tokio::test]
async fn tools_are_encoded_into_wire_format() {
    let transport = FakeTransport::with_json(chat_wire_response("ok"));
    let model = OpenAiCompatible::new(fake_provider(transport.clone()));

    model
        .generate(
            GenerateRequest::new("m", vec![Message::user("hi")]).with_tools(Some(vec![ToolDef {
                name: "read_file".to_string(),
                description: "read a file".to_string(),
                parameters: json!({ "type": "object" }),
            }])),
        )
        .await
        .unwrap();

    let body = transport.first_body();
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["function"]["name"], "read_file");
    assert_eq!(body["tools"][0]["function"]["description"], "read a file");
}

// ============================================================================
// OpenAI protocol types & conversion
// ============================================================================

#[test]
fn tool_call_fragments_merge_by_index() {
    let mut accumulated = OpenAiToolCall {
        id: "call_1".to_string(),
        call_type: Some("function".to_string()),
        index: Some(0),
        function: Some(ToolCallFunction {
            name: "read_file".to_string(),
            arguments: String::new(),
        }),
        ..Default::default()
    };

    let fragment_a = OpenAiToolCall {
        index: Some(0),
        function: Some(ToolCallFunction {
            name: String::new(),
            arguments: "{\"path\":".to_string(),
        }),
        ..Default::default()
    };
    let fragment_b = OpenAiToolCall {
        index: Some(0),
        function: Some(ToolCallFunction {
            name: String::new(),
            arguments: "\"a.txt\"}".to_string(),
        }),
        ..Default::default()
    };

    assert!(accumulated.same_call(&fragment_a));
    accumulated.merge_from(&fragment_a);
    accumulated.merge_from(&fragment_b);

    assert_eq!(
        accumulated.function.unwrap().arguments,
        "{\"path\":\"a.txt\"}"
    );
}

#[test]
fn protocol_tool_call_maps_to_domain_with_parsed_arguments() {
    let call = OpenAiToolCall {
        id: "call_1".to_string(),
        call_type: Some("function".to_string()),
        index: Some(0),
        function: Some(ToolCallFunction {
            name: "read_file".to_string(),
            arguments: "{\"path\":\"a.txt\"}".to_string(),
        }),
        name: None,
        arguments: None,
    };

    assert_eq!(
        ToolCall::from(call),
        ToolCall {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: json!({ "path": "a.txt" }),
        }
    );
}

#[test]
fn malformed_tool_arguments_fall_back_to_null() {
    let call = OpenAiToolCall {
        id: "call_1".to_string(),
        function: Some(ToolCallFunction {
            name: "read_file".to_string(),
            arguments: "{\"path\":".to_string(),
        }),
        ..Default::default()
    };

    assert_eq!(ToolCall::from(call).arguments, Value::Null);
}

#[test]
fn domain_tool_call_encodes_back_to_nested_protocol_shape() {
    let call = ToolCall {
        id: "call_1".to_string(),
        name: "read_file".to_string(),
        arguments: json!({ "path": "a.txt" }),
    };

    let encoded = serde_json::to_value(OpenAiToolCall::from(&call)).unwrap();

    assert_eq!(encoded["id"], "call_1");
    assert_eq!(encoded["type"], "function");
    assert_eq!(encoded["function"]["name"], "read_file");
    assert_eq!(encoded["function"]["arguments"], "{\"path\":\"a.txt\"}");
    assert!(encoded.get("index").is_none());
}

#[test]
fn reasoning_accepts_openrouter_and_deepseek_responses() {
    for field in ["reasoning", "reasoning_content"] {
        let mut message = json!({ "role": "assistant", "content": "answer" });
        message[field] = json!("reasoning text");
        let full: ChoiceMessage = serde_json::from_value(message.clone()).unwrap();
        let delta: ChunkDelta = serde_json::from_value(message).unwrap();
        assert_eq!(full.reasoning_content.as_deref(), Some("reasoning text"));
        assert_eq!(delta.reasoning_content.as_deref(), Some("reasoning text"));
    }
    let delta: ChunkDelta = serde_json::from_value(json!({"reasoning": null})).unwrap();
    assert!(delta.reasoning_content.is_none());
}

#[test]
fn test_response_allows_null_content_for_images() {
    let body = r#"{
        "id": "resp_123",
        "object": "chat.completion",
        "created": 1743916800,
        "model": "black-forest-labs/flux.2-klein-4b",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.png"
                            }
                        }
                    ]
                },
                "finish_reason": "stop"
            }
        ]
    }"#;

    let response: ChatCompletionResponse = serde_json::from_str(body).unwrap();
    let choice = response.choices.into_iter().next().unwrap();

    assert_eq!(choice.message.content, None);
    assert_eq!(
        choice.message.images.unwrap()[0].image_url.url,
        "https://example.com/image.png"
    );
}

#[test]
fn test_response_allows_float_u32_fields() {
    let body = r#"{
        "id": "resp_123",
        "object": "chat.completion",
        "created": 1743916800,
        "model": "black-forest-labs/flux.2-klein-4b",
        "choices": [
            {
                "index": 14417.92,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.png"
                            }
                        }
                    ]
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 1.2,
            "completion_tokens": "2.7",
            "total_tokens": 3.9,
            "prompt_cache_hit_tokens": "4",
            "prompt_cache_miss_tokens": null
        }
    }"#;

    let response: ChatCompletionResponse = serde_json::from_str(body).unwrap();
    let choice = response.choices.into_iter().next().unwrap();
    let usage = response.usage.unwrap();

    assert_eq!(choice.index, 14418);
    assert_eq!(usage.prompt_tokens, 1);
    assert_eq!(usage.completion_tokens, 3);
    assert_eq!(usage.total_tokens, Some(4));
    assert_eq!(usage.prompt_cache_hit_tokens, Some(4));
    assert_eq!(usage.prompt_cache_miss_tokens, None);
}

#[test]
fn test_stream_response_allows_audio_delta() {
    let body = r#"{
        "id": "chunk_123",
        "object": "chat.completion.chunk",
        "created": 1743916800,
        "model": "google/lyria-3-clip-preview",
        "system_fingerprint": null,
        "choices": [
            {
                "index": 0,
                "delta": {
                    "audio": {
                        "data": "YWJj",
                        "transcript": "hello"
                    }
                },
                "finish_reason": null
            }
        ]
    }"#;

    let response: ChatCompletionChunk = serde_json::from_str(body).unwrap();
    let audio = response.choices[0].delta.audio.as_ref().unwrap();

    assert_eq!(audio.data.as_deref(), Some("YWJj"));
    assert_eq!(audio.transcript.as_deref(), Some("hello"));
}

#[test]
fn generate_request_maps_domain_fields_to_chat_completion_request() {
    let request = GenerateRequest::new("deepseek-v4-pro", vec![Message::user("hello")])
        .with_temperature(0.25)
        .with_max_tokens(128)
        .with_output(OutputSpec::json_object())
        .with_reasoning(ReasoningConfig::Effort {
            level: ReasoningEffort::High,
        })
        .with_tools(Some(vec![ToolDef {
            name: "read_file".to_string(),
            description: "read".to_string(),
            parameters: json!({ "type": "object" }),
        }]));

    let body = serde_json::to_value(ChatCompletionRequest::try_from(&request).unwrap()).unwrap();

    assert_eq!(body["model"], "deepseek-v4-pro");
    assert_eq!(body["temperature"], 0.25);
    assert_eq!(body["max_tokens"], 128);
    assert_eq!(body["response_format"]["type"], "json_object");
    assert_eq!(body["reasoning_effort"], "high");
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["function"]["name"], "read_file");
    assert!(body.get("stream").is_none());
    assert!(body.get("stream_options").is_none());
    assert!(body.get("modalities").is_none());
}

// ============================================================================
// OpenRouter image protocol & types
// ============================================================================

#[tokio::test]
async fn generate_uses_dedicated_images_endpoint_and_returns_media_source() {
    let transport = FakeTransport::with_json(json!({
        "data": [{ "b64_json": "aW1hZ2U=", "media_type": "image/webp" }]
    }));
    let model = OpenRouterImages::new(fake_provider(transport.clone()));

    let response = model
        .generate(GenerateRequest::image_with(
            "krea/krea-2-medium-turbo",
            "cinematic landscape",
            ImageOptions::new()
                .with_resolution("1K")
                .with_aspect_ratio("16:9"),
        ))
        .await
        .unwrap();

    assert_eq!(
        response.message.images(),
        vec![&MediaSource::base64("image/webp", "aW1hZ2U=")]
    );
    assert_eq!(transport.recorded_paths(), vec!["/images"]);
    let body = transport.first_body();
    assert_eq!(body["model"], "krea/krea-2-medium-turbo");
    assert_eq!(body["prompt"], "cinematic landscape");
    assert_eq!(body["resolution"], "1K");
    assert_eq!(body["aspect_ratio"], "16:9");
    assert!(body.get("messages").is_none());
    assert!(body.get("size").is_none());
}

#[tokio::test]
async fn generate_rejects_empty_prompt_without_calling_transport() {
    let transport = FakeTransport::with_json(json!({ "data": [] }));
    let model = OpenRouterImages::new(fake_provider(transport.clone()));

    let error = model
        .generate(GenerateRequest::image("m", "   "))
        .await
        .expect_err("empty prompt should be rejected");

    assert!(matches!(error, ModelError::InvalidRequest(_)));
    assert!(transport.recorded().is_empty());
}

#[test]
fn generated_image_prefers_url_and_maps_b64_to_media_source() {
    assert_eq!(
        generated_image_to_source(GeneratedImage {
            url: Some("https://example.com/a.png".to_string()),
            b64_json: Some("ignored".to_string()),
            media_type: None,
        }),
        Some(MediaSource::url("https://example.com/a.png"))
    );
    assert_eq!(
        generated_image_to_source(GeneratedImage {
            url: None,
            b64_json: Some("aW1hZ2U=".to_string()),
            media_type: Some("image/webp".to_string()),
        }),
        Some(MediaSource::base64("image/webp", "aW1hZ2U="))
    );
    assert_eq!(
        generated_image_to_source(GeneratedImage {
            url: None,
            b64_json: None,
            media_type: None,
        }),
        None
    );
}
