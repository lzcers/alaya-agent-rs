//! SSE 分帧。
//!
//! 只负责 HTTP 规范里的「事件切分」：把字节流切成一个个事件的 `data` 载荷。
//! 载荷本身是什么（JSON、`[DONE]`、别的文本）由协议层决定，传输层不解释。

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

/// 从缓冲区中切出所有完整事件的 `data` 载荷，未完成的事件留在缓冲区里。
///
/// 多个 `data:` 行按 SSE 规范用 `\n` 拼接；纯空白事件被丢弃。
pub(crate) fn drain_events(buffer: &mut Vec<u8>) -> Vec<String> {
    normalize_crlf(buffer);

    let mut events = Vec::new();
    while let Some(idx) = buffer.windows(2).position(|window| window == b"\n\n") {
        let remaining = buffer.split_off(idx + 2);
        buffer.truncate(idx);
        let frame = std::mem::replace(buffer, remaining);

        let payload = String::from_utf8_lossy(&frame)
            .lines()
            .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
            .collect::<Vec<_>>()
            .join("\n");

        if !payload.is_empty() {
            events.push(payload);
        }
    }
    events
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn handles_split_chunks() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"data: {\"id\":\"abc\"");
        assert!(drain_events(&mut buffer).is_empty());

        buffer.extend_from_slice(b",\"object\":\"chat.completion.chunk\"}\n\n");
        let events = drain_events(&mut buffer);

        assert_eq!(events, vec!["{\"id\":\"abc\",\"object\":\"chat.completion.chunk\"}"]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn preserves_multibyte_payload_across_every_chunk_boundary() {
        let payload = json!({ "content": "铜片🙂" }).to_string();
        let frame = format!("data: {payload}\r\n\r\n");

        for split_at in 0..=frame.len() {
            let mut buffer = Vec::new();
            buffer.extend_from_slice(&frame.as_bytes()[..split_at]);
            let mut events = drain_events(&mut buffer);

            buffer.extend_from_slice(&frame.as_bytes()[split_at..]);
            events.extend(drain_events(&mut buffer));

            assert_eq!(events.len(), 1, "split_at={split_at}");
            assert_eq!(events[0], payload, "split_at={split_at}");
            assert!(buffer.is_empty(), "split_at={split_at}");
        }
    }

    #[test]
    fn joins_multiple_data_lines_and_keeps_done_marker() {
        let mut buffer = b"data: {\"a\":1}\ndata: {\"b\":2}\n\ndata: [DONE]\n\n".to_vec();
        let events = drain_events(&mut buffer);

        assert_eq!(events, vec!["{\"a\":1}\n{\"b\":2}", "[DONE]"]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn ignores_whitespace_only_events() {
        let mut buffer = b"\n\ndata: {\"a\":1}\n\n".to_vec();

        assert_eq!(drain_events(&mut buffer), vec!["{\"a\":1}"]);
    }
}
