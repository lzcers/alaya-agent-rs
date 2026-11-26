//! SSE 分帧。
//!
//! 负责 HTTP 规范里的「事件切分」：把字节流切成一个个包含 `event` / `data` / `id`
//! 的标准 [`SseEvent`]。载荷本身是什么（JSON、`[DONE]`、纯文本）由协议层决定，
//! 传输层不做业务解释。

use futures::StreamExt;

use super::{BodyStream, TransportError};

/// 一条标准 SSE 事件。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
}

/// 把 CRLF 归一化为 LF，避免逐字节切分时把 `\r\n\r\n` 误判。
fn normalize_crlf(buffer: &mut Vec<u8>) {
    let mut read = 0;
    let mut write = 0;

    while read < buffer.len() {
        if buffer[read] == b'\r' && buffer.get(read + 1) == Some(&b'\n') {
            buffer[write] = b'\n';
            read += 2;
        } else {
            buffer[write] = buffer[read];
            read += 1;
        }
        write += 1;
    }

    buffer.truncate(write);
}

/// 按 WHATWG SSE 规范剥离字段名冒号后的单个可选空格。
fn strip_field_space(value: &str) -> &str {
    value.strip_prefix(' ').unwrap_or(value)
}

/// 从缓冲区中切出所有完整 SSE 事件，未完成的事件留在缓冲区里。
///
/// 多个 `data:` 行按规范用 `\n` 拼接；注释行（`:` 开头）与空 `data` 事件被丢弃。
pub(crate) fn drain_events(buffer: &mut Vec<u8>) -> Vec<SseEvent> {
    normalize_crlf(buffer);

    let mut events = Vec::new();
    while let Some(idx) = buffer.windows(2).position(|window| window == b"\n\n") {
        let remaining = buffer.split_off(idx + 2);
        buffer.truncate(idx);
        let frame = std::mem::replace(buffer, remaining);

        let mut event_name: Option<String> = None;
        let mut id: Option<String> = None;
        let mut data_lines: Vec<&str> = Vec::new();

        let text = String::from_utf8_lossy(&frame);
        for line in text.lines() {
            if line.is_empty() || line.starts_with(':') {
                continue;
            }
            if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(strip_field_space(rest));
            } else if let Some(rest) = line.strip_prefix("event:") {
                let value = strip_field_space(rest);
                if !value.is_empty() {
                    event_name = Some(value.to_string());
                }
            } else if let Some(rest) = line.strip_prefix("id:") {
                let value = strip_field_space(rest);
                if !value.is_empty() {
                    id = Some(value.to_string());
                }
            }
        }

        if !data_lines.is_empty() {
            let data = data_lines.join("\n");
            if !data.is_empty() {
                events.push(SseEvent {
                    event: event_name,
                    data,
                    id,
                });
            }
        }
    }
    events
}

/// 把响应体字节流切成标准 SSE 事件流。
///
/// 中途的读取失败以 `Err` 形式出现在流里并终止流。
pub fn events(
    mut body: BodyStream,
) -> impl futures::Stream<Item = Result<SseEvent, TransportError>> {
    async_stream::stream! {
        let mut buffer = Vec::new();

        while let Some(chunk) = body.next().await {
            match chunk {
                Ok(bytes) => {
                    buffer.extend_from_slice(&bytes);
                    for event in drain_events(&mut buffer) {
                        yield Ok(event);
                    }
                }
                Err(error) => {
                    yield Err(error);
                    return;
                }
            }
        }

        // 有些服务端最后一条事件不带结尾空行，补一个终止符把残留冲刷出来。
        buffer.extend_from_slice(b"\n\n");
        for event in drain_events(&mut buffer) {
            yield Ok(event);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytes::Bytes;
    use serde_json::json;

    #[test]
    fn handles_split_chunks() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"data: {\"id\":\"abc\"");
        assert!(drain_events(&mut buffer).is_empty());

        buffer.extend_from_slice(b",\"object\":\"chat.completion.chunk\"}\n\n");
        let parsed = drain_events(&mut buffer);

        assert_eq!(
            parsed,
            vec![SseEvent {
                event: None,
                data: "{\"id\":\"abc\",\"object\":\"chat.completion.chunk\"}".to_string(),
                id: None,
            }]
        );
        assert!(buffer.is_empty());
    }

    #[test]
    fn preserves_multibyte_payload_across_every_chunk_boundary() {
        let payload = json!({ "content": "铜片🙂" }).to_string();
        let frame = format!("event: delta\nid: 42\ndata: {payload}\r\n\r\n");

        for split_at in 0..=frame.len() {
            let mut buffer = Vec::new();
            buffer.extend_from_slice(&frame.as_bytes()[..split_at]);
            let mut parsed = drain_events(&mut buffer);

            buffer.extend_from_slice(&frame.as_bytes()[split_at..]);
            parsed.extend(drain_events(&mut buffer));

            assert_eq!(parsed.len(), 1, "split_at={split_at}");
            assert_eq!(parsed[0].event.as_deref(), Some("delta"));
            assert_eq!(parsed[0].id.as_deref(), Some("42"));
            assert_eq!(parsed[0].data, payload, "split_at={split_at}");
            assert!(buffer.is_empty(), "split_at={split_at}");
        }
    }

    #[test]
    fn joins_multiple_data_lines_and_keeps_done_marker() {
        let mut buffer = b"data: {\"a\":1}\ndata: {\"b\":2}\n\ndata: [DONE]\n\n".to_vec();
        let data: Vec<String> = drain_events(&mut buffer)
            .into_iter()
            .map(|e| e.data)
            .collect();

        assert_eq!(data, vec!["{\"a\":1}\n{\"b\":2}", "[DONE]"]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn ignores_whitespace_and_comment_events() {
        let mut buffer = b": keep-alive\n\n\n\ndata: {\"a\":1}\n\n".to_vec();
        let data: Vec<String> = drain_events(&mut buffer)
            .into_iter()
            .map(|e| e.data)
            .collect();

        assert_eq!(data, vec!["{\"a\":1}"]);
    }

    #[tokio::test]
    async fn events_frames_a_chunked_body() {
        let body: BodyStream = Box::pin(futures::stream::iter(vec![
            Ok(Bytes::from_static(b"data: {\"i\":1}\n\ndata: {\"i\":")),
            Ok(Bytes::from_static(b"2}\n\ndata: [DONE]\n\n")),
        ]));

        let parsed: Vec<String> = events(body)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .map(|e| e.data)
            .collect();

        assert_eq!(parsed, vec!["{\"i\":1}", "{\"i\":2}", "[DONE]"]);
    }

    #[tokio::test]
    async fn events_flushes_trailing_event_without_blank_line() {
        let body: BodyStream = Box::pin(futures::stream::once(async {
            Ok(Bytes::from_static(b"data: {\"last\":true}"))
        }));

        let parsed: Vec<String> = events(body)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
            .into_iter()
            .map(|e| e.data)
            .collect();

        assert_eq!(parsed, vec!["{\"last\":true}"]);
    }
}
