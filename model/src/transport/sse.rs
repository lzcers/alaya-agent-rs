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
fn drain_events(buffer: &mut Vec<u8>) -> Vec<SseEvent> {
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
