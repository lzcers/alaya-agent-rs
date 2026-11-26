use crate::agent::{ToolCall, ToolExecutor};
use crate::capability::{
    FinishReason, GenerateCapability, GenerateRequest, PartDelta, StreamAccumulator, StreamEvent,
    format_model_error,
};
use crate::{ContentPart, Message, Usage};
use async_stream::stream;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

/// 工具执行结果
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CallToolResult {
    /// 工具调用 ID
    pub call_id: String,
    /// 工具名称
    pub tool_name: String,
    /// 执行是否成功
    pub success: bool,
    /// 输出内容（JSON 字符串）
    pub output: String,
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum CallToolError {
    #[error("call tool timeout.")]
    Timeout,
}

/// 调用模型事件（纯数据，支持多模态流式输出）
#[derive(Debug, Clone)]
pub enum CallModelEvent {
    /// 文本增量片段
    TextChunk(String),
    /// 推理增量片段
    ReasoningChunk(String),
    /// 音频增量片段
    AudioChunk {
        data: Option<String>,
        transcript: Option<String>,
        format: Option<String>,
    },
    /// 已产出的完整内容块（工具调用、图片、文件等）
    PartCompleted(ContentPart),
    /// 模型响应完整结束
    Completed {
        message: Message,
        usage: Option<Usage>,
        finish_reason: Option<FinishReason>,
    },
    /// 调用或流式传输错误
    Error(String),
}

/// 调用具备统一生成能力的模型，流式转发增量并归并完整的 [`Message`]。
pub fn call_model(
    model: &(dyn GenerateCapability + Sync),
    request: GenerateRequest,
) -> impl Stream<Item = CallModelEvent> {
    stream! {
        let mut response_stream = match model.generate_stream(request).await {
            Ok(s) => s,
            Err(e) => {
                yield CallModelEvent::Error(format_model_error(&e));
                return;
            }
        };

        let mut acc = StreamAccumulator::new();
        while let Some(item) = response_stream.next().await {
            let event = match item {
                Ok(event) => event,
                Err(e) => {
                    yield CallModelEvent::Error(format_model_error(&e));
                    return;
                }
            };

            acc.push(&event);

            match event {
                StreamEvent::Delta(PartDelta::Text { text }) if !text.is_empty() => {
                    yield CallModelEvent::TextChunk(text);
                }
                StreamEvent::Delta(PartDelta::Reasoning { text }) if !text.is_empty() => {
                    yield CallModelEvent::ReasoningChunk(text);
                }
                StreamEvent::Delta(PartDelta::Audio {
                    data,
                    transcript,
                    format,
                }) => {
                    yield CallModelEvent::AudioChunk {
                        data,
                        transcript,
                        format,
                    };
                }
                StreamEvent::Part(part) => {
                    yield CallModelEvent::PartCompleted(part);
                }
                StreamEvent::Delta(_) | StreamEvent::Usage(_) | StreamEvent::Finished { .. } => {}
            }
        }

        let response = acc.finish();
        yield CallModelEvent::Completed {
            message: response.message,
            usage: response.usage,
            finish_reason: response.finish_reason,
        };
    }
}

pub async fn call_tool(tool_executor: &dyn ToolExecutor, call: &ToolCall) -> CallToolResult {
    let call_id = call.id.clone();
    let tool_name = call.name.clone();

    let result = tool_executor.execute(call).await;

    let (success, output) = match result {
        Ok(r) if r.success => {
            let output = serde_json::to_string(&r.output).unwrap_or_else(|_| "{}".to_string());
            (true, output)
        }
        Ok(r) => {
            let error_msg = serde_json::to_string(&serde_json::json!({
                "error": r.output
            }))
            .unwrap_or_else(|_| r#"{\"error\": \"Tool execution failed\"}"#.to_string());
            (false, error_msg)
        }
        Err(e) => {
            let error_msg = serde_json::to_string(&serde_json::json!({
                "error": e.to_string()
            }))
            .unwrap_or_else(|_| r#"{\"error\": \"Tool execution error\"}"#.to_string());
            (false, error_msg)
        }
    };

    CallToolResult {
        call_id,
        tool_name,
        success,
        output,
    }
}

/// 并行执行多个工具
pub fn call_tools(
    tool_executor: &dyn ToolExecutor,
    tools_call: &[ToolCall],
) -> impl Stream<Item = CallToolResult> {
    let futures: Vec<_> = tools_call
        .iter()
        .map(|call| call_tool(tool_executor, call))
        .collect();
    stream! {
        let results = futures::future::join_all(futures).await;
        for result in results {
            yield result;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::{GenerateResponse, GenerateStream, ModelError};
    use async_trait::async_trait;
    use futures::{StreamExt, stream};

    struct MockModel {
        events: Vec<StreamEvent>,
    }

    #[async_trait]
    impl GenerateCapability for MockModel {
        async fn generate(&self, _request: GenerateRequest) -> Result<GenerateResponse, ModelError> {
            panic!("generate should not be called in this test");
        }

        async fn generate_stream(
            &self,
            _request: GenerateRequest,
        ) -> Result<GenerateStream, ModelError> {
            let items: Vec<Result<StreamEvent, ModelError>> =
                self.events.iter().cloned().map(Ok).collect();
            Ok(Box::pin(stream::iter(items)))
        }
    }

    #[tokio::test]
    async fn call_model_captures_usage_emitted_after_finished_event() {
        let model = MockModel {
            events: vec![
                StreamEvent::Delta(PartDelta::Text {
                    text: "hello".to_string(),
                }),
                StreamEvent::Finished {
                    reason: Some(FinishReason::Stop),
                },
                StreamEvent::Usage(Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    total_tokens: 15,
                    ..Usage::default()
                }),
            ],
        };

        let events = call_model(
            &model,
            GenerateRequest::new("mock-model", vec![Message::user("hi")]),
        )
        .collect::<Vec<_>>()
        .await;

        assert!(matches!(
            events.as_slice(),
            [
                CallModelEvent::TextChunk(text),
                CallModelEvent::Completed {
                    message,
                    usage: Some(usage),
                    finish_reason: Some(FinishReason::Stop),
                }
            ] if text == "hello" && message.text() == "hello" && usage.total_tokens == 15
        ));
    }
}
