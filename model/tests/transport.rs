use std::time::Duration;

use alaya_model::transport::{BodyStream, HttpRequest, HttpResponse, HttpTransport, SseEvent, Transport, sse};
use bytes::Bytes;
use futures::StreamExt;
use reqwest::header::CONTENT_TYPE;
use serde_json::json;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
};

// ============================================================================
// HttpRequest / HttpResponse
// ============================================================================

#[tokio::test]
async fn json_body_sets_content_type_and_encodes() {
    let request = HttpRequest::post("https://example.com/x")
        .with_json_body(&json!({ "a": 1 }))
        .unwrap();

    assert_eq!(
        request.headers.get(CONTENT_TYPE).unwrap(),
        "application/json"
    );
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&request.body).unwrap(),
        json!({ "a": 1 })
    );
}

#[tokio::test]
async fn buffered_response_collects_once() {
    let response = HttpResponse::buffered(200, Bytes::from_static(b"{\"ok\":true}"));

    assert!(response.is_success());
    assert_eq!(
        response.json::<serde_json::Value>().await.unwrap()["ok"],
        true
    );
}

#[tokio::test]
async fn binary_body_is_expressible() {
    let request = HttpRequest::post("https://example.com/speech")
        .with_bytes_body("audio/mpeg", Bytes::from_static(b"\xff\xfb\x90"))
        .unwrap();

    assert_eq!(request.body.len(), 3);
    assert_eq!(request.headers.get(CONTENT_TYPE).unwrap(), "audio/mpeg");
}

// ============================================================================
// SSE
// ============================================================================

async fn collect_sse(chunks: Vec<Bytes>) -> Vec<SseEvent> {
    let body: BodyStream = Box::pin(futures::stream::iter(chunks.into_iter().map(Ok)));
    sse::events(body)
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

#[tokio::test]
async fn handles_split_chunks() {
    let parsed = collect_sse(vec![
        Bytes::from_static(b"data: {\"id\":\"abc\""),
        Bytes::from_static(b",\"object\":\"chat.completion.chunk\"}\n\n"),
    ])
    .await;

    assert_eq!(
        parsed,
        vec![SseEvent {
            event: None,
            data: "{\"id\":\"abc\",\"object\":\"chat.completion.chunk\"}".to_string(),
            id: None,
        }]
    );
}

#[tokio::test]
async fn preserves_multibyte_payload_across_every_chunk_boundary() {
    let payload = json!({ "content": "铜片🙂" }).to_string();
    let frame = format!("event: delta\nid: 42\ndata: {payload}\r\n\r\n");

    for split_at in 0..=frame.len() {
        let parsed = collect_sse(vec![
            Bytes::copy_from_slice(&frame.as_bytes()[..split_at]),
            Bytes::copy_from_slice(&frame.as_bytes()[split_at..]),
        ])
        .await;

        assert_eq!(parsed.len(), 1, "split_at={split_at}");
        assert_eq!(parsed[0].event.as_deref(), Some("delta"));
        assert_eq!(parsed[0].id.as_deref(), Some("42"));
        assert_eq!(parsed[0].data, payload, "split_at={split_at}");
    }
}

#[tokio::test]
async fn joins_multiple_data_lines_and_keeps_done_marker() {
    let data: Vec<String> = collect_sse(vec![Bytes::from_static(
        b"data: {\"a\":1}\ndata: {\"b\":2}\n\ndata: [DONE]\n\n",
    )])
    .await
    .into_iter()
    .map(|e| e.data)
    .collect();

    assert_eq!(data, vec!["{\"a\":1}\n{\"b\":2}", "[DONE]"]);
}

#[tokio::test]
async fn ignores_whitespace_and_comment_events() {
    let data: Vec<String> = collect_sse(vec![Bytes::from_static(
        b": keep-alive\n\n\n\ndata: {\"a\":1}\n\n",
    )])
    .await
    .into_iter()
    .map(|e| e.data)
    .collect();

    assert_eq!(data, vec!["{\"a\":1}"]);
}

#[tokio::test]
async fn events_frames_a_chunked_body() {
    let parsed: Vec<String> = collect_sse(vec![
        Bytes::from_static(b"data: {\"i\":1}\n\ndata: {\"i\":"),
        Bytes::from_static(b"2}\n\ndata: [DONE]\n\n"),
    ])
    .await
    .into_iter()
    .map(|e| e.data)
    .collect();

    assert_eq!(parsed, vec!["{\"i\":1}", "{\"i\":2}", "[DONE]"]);
}

#[tokio::test]
async fn events_flushes_trailing_event_without_blank_line() {
    let parsed: Vec<String> =
        collect_sse(vec![Bytes::from_static(b"data: {\"last\":true}")])
            .await
            .into_iter()
            .map(|e| e.data)
            .collect();

    assert_eq!(parsed, vec!["{\"last\":true}"]);
}

// ============================================================================
// HttpTransport
// ============================================================================

#[test]
fn transport_holds_connection_config_only() {
    let transport = HttpTransport::new()
        .with_proxy("http://127.0.0.1:7890")
        .with_timeout(Duration::from_secs(60));

    assert_eq!(transport.proxy_url(), Some("http://127.0.0.1:7890"));
    assert_eq!(transport.timeout(), Duration::from_secs(60));
}

#[tokio::test]
async fn send_posts_bytes_and_returns_buffered_json() {
    let (base_url, server) =
        serve_once("HTTP/1.1 200 OK", &json!({ "ok": true }).to_string()).await;
    let transport = HttpTransport::new();

    let response = transport
        .send(
            HttpRequest::post(format!("{base_url}/chat/completions"))
                .with_header("x-test", "1")
                .unwrap()
                .with_json_body(&json!({ "model": "m" }))
                .unwrap(),
        )
        .await
        .expect("request should succeed");

    let raw = server.await.expect("test server should complete");
    let (headers, body) = raw.split_once("\r\n\r\n").expect("headers and body");

    assert!(headers.starts_with("POST /chat/completions HTTP/1.1"));
    assert!(headers.to_ascii_lowercase().contains("x-test: 1"));
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(body).unwrap(),
        json!({ "model": "m" })
    );
    assert!(response.is_success());
    assert_eq!(
        response.json::<serde_json::Value>().await.unwrap()["ok"],
        true
    );
}

#[tokio::test]
async fn send_returns_raw_bytes_for_binary_responses() {
    let (base_url, server) = serve_once("HTTP/1.1 200 OK", "not-json-at-all").await;
    let transport = HttpTransport::new();

    let response = transport
        .send(HttpRequest::post(format!("{base_url}/audio/speech")))
        .await
        .expect("request should succeed");
    let _ = server.await;

    assert_eq!(
        response.bytes().await.unwrap(),
        Bytes::from_static(b"not-json-at-all")
    );
}

#[tokio::test]
async fn send_does_not_fail_on_non_success_status() {
    let (base_url, server) = serve_once("HTTP/1.1 429 Too Many Requests", "slow down").await;
    let transport = HttpTransport::new();

    let response = transport
        .send(HttpRequest::post(format!("{base_url}/x")))
        .await
        .expect("transport should return the response, not an error");
    let _ = server.await;

    assert_eq!(response.status(), 429);
    assert!(!response.is_success());
}

async fn serve_once(
    status_line: &str,
    response_body: &str,
) -> (String, tokio::task::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("test listener should bind");
    let address = listener.local_addr().expect("listener address");
    let status_line = status_line.to_string();
    let response = format!(
        "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{response_body}",
        response_body.len()
    );

    let handle = tokio::spawn(async move {
        let (mut socket, _) = listener
            .accept()
            .await
            .expect("test server should accept a connection");
        let mut request = Vec::new();
        let mut expected_len = None;

        loop {
            let mut chunk = [0_u8; 4096];
            let bytes_read = socket
                .read(&mut chunk)
                .await
                .expect("test server should read request");
            if bytes_read == 0 {
                break;
            }
            request.extend_from_slice(&chunk[..bytes_read]);

            if expected_len.is_none()
                && let Some(header_end) = request.windows(4).position(|part| part == b"\r\n\r\n")
            {
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        let (name, value) = line.split_once(':')?;
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or_default();
                expected_len = Some(header_end + 4 + content_length);
            }

            if expected_len.is_some_and(|length| request.len() >= length) {
                break;
            }
        }

        socket
            .write_all(response.as_bytes())
            .await
            .expect("test server should write response");

        String::from_utf8(request).expect("request should be UTF-8")
    });

    (format!("http://{address}"), handle)
}
