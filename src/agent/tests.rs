use crate::agent::tools::playwright_cli::PlaywrightCliTool;
use crate::agent::{
    AgentActor, AgentActorEvent, AgentTerminalReason, Context, GenericToolExecutor, StepResult,
    Tool, ToolCall, ToolDef, ToolExecutorError,
};

use crate::capability::{
    FinishReason, GenerateCapability, GenerateRequest, GenerateResponse, GenerateStream,
    ModelError, ModelProfile, PartDelta, StreamEvent,
};
use crate::endpoints::deepseek_from_env;
use crate::router::ModelRouter;
use crate::{ContentPart, Message, Usage};
use async_trait::async_trait;
use futures::stream;
use serde_json::{Value, json};
use std::sync::Arc;
use tokio::sync::mpsc;

fn print_test_banner(title: &str) {
    eprintln!("\n=== {} ===", title);
}

fn print_section(title: &str) {
    eprintln!("\n[{}]", title);
}

fn print_field(label: &str, value: impl std::fmt::Display) {
    eprintln!("{}: {}", label, value);
}

fn format_json_value(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let mut iter = text.chars();
    let preview: String = iter.by_ref().take(max_chars).collect();
    if iter.next().is_some() {
        format!("{}...", preview)
    } else {
        preview
    }
}

fn print_tool_call(index: usize, call: &ToolCall) {
    let name = call.name.clone();
    print_field(
        &format!("tool_{}_name", index),
        if name.is_empty() {
            "<unknown>".to_string()
        } else {
            name
        },
    );
    print_field(
        &format!("tool_{}_args", index),
        preview_text(&format_json_value(&call.arguments), 220),
    );
}

fn print_tool_result(call_id: &str, success: bool, output: &str) {
    print_section("Tool Result");
    print_field("call_id", call_id);
    print_field("success", success);
    print_field("output", preview_text(output, 220));
}

fn print_step_snapshot(message: &Message) {
    print_section("Step Completed");
    let text = message.text();
    if !text.is_empty() {
        print_field("content", preview_text(&text, 220));
    }
    if let Some(reasoning) = message.reasoning().filter(|text| !text.is_empty()) {
        print_field("reasoning", preview_text(&reasoning, 220));
    }
    print_field("tool_calls", message.tool_calls().len());
}

fn print_step_finalized(step_index: usize, result: &StepResult, frame: &impl std::fmt::Debug) {
    print_section(&format!("Step Finalized ({})", step_index));
    let status = match result {
        StepResult::Continue { .. } => "continue",
        StepResult::Done { .. } => "done",
        StepResult::Error(_) => "error",
    };
    print_field("result", status);
    print_field("frame", preview_text(&format!("{:?}", frame), 220));
}

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

struct EchoTool {
    def: ToolDef,
}

impl EchoTool {
    fn new() -> Self {
        Self {
            def: ToolDef {
                name: "echo".to_string(),
                description: "Echo input arguments".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "value": { "type": "string" }
                    },
                    "required": ["value"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for EchoTool {
    fn definition(&self) -> &ToolDef {
        &self.def
    }

    async fn execute(&self, arguments: Value) -> Result<Value, ToolExecutorError> {
        Ok(arguments)
    }
}

#[tokio::test]
async fn test_agent_actor_with_deepseek_and_playwright() {
    dotenv::dotenv().ok();

    print_test_banner("AgentActor DeepSeek + Playwright Test");

    let provider = match deepseek_from_env() {
        Ok(p) => Arc::new(p),
        Err(_) => {
            print_section("Skipped");
            print_field("reason", "未设置 DEEPSEEK_API_KEY 环境变量");
            return;
        }
    };

    let mut model = ModelRouter::new();
    model.register(
        "deepseek-reasoner",
        ModelProfile::text().with_reasoning(true),
        provider,
    );

    let mut executor = GenericToolExecutor::new();
    executor.register(PlaywrightCliTool::new());

    let mut context = Context::new();
    context.add_message(Message::system(
        r#"你是一个聪明的助手，当用户需要访问网页或提取网页内容时, 你可以使用工具完成任务。"#,
    ));
    context.add_message(Message::user(
        "请帮我总结 https://www.peopleapp.com/column/30051629695-500007391518 网页的内容",
    ));

    print_section("User Request");
    print_field(
        "prompt",
        "请帮我总结 https://www.peopleapp.com/column/30051629695-500007391518 网页的内容",
    );

    let actor = AgentActor::new(
        model,
        GenerateRequest::new("deepseek-reasoner", Vec::new()),
        executor,
        context,
    );

    let mut handle = actor.run_loop();

    let mut events: Vec<AgentActorEvent> = Vec::new();
    let mut tool_calls_log: Vec<String> = Vec::new();
    let mut step_index = 0usize;

    while let Some(event) = handle.event_rx.recv().await {
        match &event {
            AgentActorEvent::ContentChunk(_)
            | AgentActorEvent::ReasoningChunk(_)
            | AgentActorEvent::AudioChunk { .. }
            | AgentActorEvent::PartCompleted(_) => {}
            AgentActorEvent::ToolCalls(calls) => {
                print_section(&format!("Tool Calls ({})", calls.len()));
                for (index, call) in calls.iter().enumerate() {
                    tool_calls_log.push(call.name.clone());
                    print_tool_call(index + 1, call);
                }
            }
            AgentActorEvent::ToolResult {
                call_id,
                success,
                output,
            } => {
                print_tool_result(call_id, *success, output);
            }
            AgentActorEvent::StepCompleted { message } => {
                print_step_snapshot(message);
            }
            AgentActorEvent::StepFinalized { result, frame } => {
                step_index += 1;
                print_step_finalized(step_index, result, frame);
            }
            AgentActorEvent::Iteration {
                iteration,
                message_count,
            } => {
                print_section("Iteration Committed");
                print_field("iteration", iteration.to_string());
                print_field("messages", message_count.to_string());
            }
            AgentActorEvent::Completed => {
                print_section("Terminal Event");
                print_field("status", "completed");
            }
            AgentActorEvent::Cancelled => {
                print_section("Terminal Event");
                print_field("status", "cancelled");
            }
            AgentActorEvent::Error(e) => {
                print_section("Terminal Event");
                print_field("status", "error");
                print_field("reason", e.to_string());
            }
            AgentActorEvent::HookEvent {
                hook,
                kind,
                payload,
            } => {
                print_section("Hook Event");
                print_field("hook", hook);
                print_field("kind", kind);
                print_field("payload", preview_text(&format_json_value(payload), 220));
            }
            AgentActorEvent::MaxIterations { iteration } => {
                print_section("Terminal Event");
                print_field("status", "max_iterations");
                print_field("iteration", iteration.to_string());
            }
            AgentActorEvent::AskUser { .. } => {
                todo!();
            }
        }

        events.push(event.clone());

        if matches!(
            events.last(),
            Some(AgentActorEvent::Completed)
                | Some(AgentActorEvent::Cancelled)
                | Some(AgentActorEvent::Error(_))
                | Some(AgentActorEvent::MaxIterations { .. })
        ) {
            break;
        }
    }

    let terminal_status = events
        .iter()
        .rev()
        .find_map(|event| match event {
            AgentActorEvent::Completed => Some("completed".to_string()),
            AgentActorEvent::Cancelled => Some("cancelled".to_string()),
            AgentActorEvent::Error(error) => Some(format!("error: {}", error)),
            AgentActorEvent::MaxIterations { iteration } => {
                Some(format!("max_iterations ({})", iteration))
            }
            _ => None,
        })
        .unwrap_or_else(|| "unknown".to_string());

    let tool_call_count = if tool_calls_log.is_empty() {
        events
            .iter()
            .map(|event| match event {
                AgentActorEvent::StepFinalized {
                    result: StepResult::Continue { tools_call, .. },
                    ..
                } => tools_call.len(),
                _ => 0,
            })
            .sum()
    } else {
        tool_calls_log.len()
    };

    let mut tools_used = if tool_calls_log.is_empty() {
        events
            .iter()
            .flat_map(|event| match event {
                AgentActorEvent::StepFinalized {
                    result: StepResult::Continue { tools_call, .. },
                    ..
                } => tools_call
                    .iter()
                    .map(|call| call.name.clone())
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect::<Vec<_>>()
    } else {
        tool_calls_log.clone()
    };
    tools_used.sort();
    tools_used.dedup();

    print_section("Run Summary");
    print_field("steps", step_index.to_string());
    print_field("events", events.len().to_string());
    print_field("tool_calls", tool_call_count.to_string());
    if !tools_used.is_empty() {
        print_field("tools", tools_used.join(", "));
    }
    print_field("terminal", terminal_status);

    let finished = events.iter().any(|e| {
        matches!(
            e,
            AgentActorEvent::Completed | AgentActorEvent::MaxIterations { .. }
        )
    });
    assert!(finished, "Agent 应该正常结束执行");
}

#[tokio::test]
async fn test_agent_actor_accumulates_usage_from_stream_response() {
    let model = MockModel {
        events: vec![
            StreamEvent::Delta(PartDelta::Text {
                text: "done".to_string(),
            }),
            StreamEvent::Finished {
                reason: Some(FinishReason::Stop),
            },
            StreamEvent::Usage(Usage {
                input_tokens: 13,
                output_tokens: 8,
                total_tokens: 21,
                ..Usage::default()
            }),
        ],
    };

    let executor = GenericToolExecutor::new();
    let mut context = Context::new();
    context.add_message(Message::user("hello"));

    let mut actor = AgentActor::new(
        model,
        GenerateRequest::new("mock-model", Vec::new()),
        executor,
        context,
    );
    let result = actor.run_step(None).await;

    assert!(matches!(result, crate::agent::StepResult::Done { .. }));
    assert_eq!(actor.state().metrics.tokens.input_tokens, 13);
    assert_eq!(actor.state().metrics.tokens.output_tokens, 8);
    assert_eq!(actor.state().metrics.tokens.total_tokens, 21);
    assert_eq!(actor.state().metrics.execution.iteration, 1);
    assert_eq!(
        actor.state().metrics.execution.terminal_reason,
        Some(AgentTerminalReason::Completed)
    );
    assert!(actor.state().metrics.timeline.started_at.is_some());
    assert!(actor.state().metrics.timeline.finished_at.is_some());
    assert!(actor.state().metrics.timeline.total_duration_ms.is_some());
}

#[tokio::test]
async fn test_agent_actor_emits_step_frame_metrics_to_upper_layer() {
    let model = MockModel {
        events: vec![
            StreamEvent::Delta(PartDelta::Text {
                text: "use tool".to_string(),
            }),
            StreamEvent::Part(ContentPart::ToolCall(ToolCall {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                arguments: json!({ "value": "ok" }),
            })),
            StreamEvent::Finished {
                reason: Some(FinishReason::ToolCalls),
            },
        ],
    };

    let mut executor = GenericToolExecutor::new();
    executor.register(EchoTool::new());

    let mut context = Context::new();
    context.add_message(Message::user("hello"));

    let mut actor = AgentActor::new(
        model,
        GenerateRequest::new("mock-model", Vec::new()),
        executor,
        context,
    );
    let (event_tx, mut event_rx) = mpsc::channel(8);
    let result = actor.run_step(Some(event_tx)).await;

    assert!(matches!(result, crate::agent::StepResult::Continue { .. }));
    assert_eq!(actor.state().metrics.execution.iteration, 1);
    assert_eq!(actor.state().metrics.tools.call_count, 1);
    assert_eq!(actor.state().metrics.tools.success_count, 1);
    assert_eq!(actor.state().metrics.tools.failure_count, 0);
    assert_eq!(actor.state().metrics.execution.terminal_reason, None);
    assert!(actor.state().metrics.timeline.started_at.is_some());
    assert!(actor.state().metrics.timeline.finished_at.is_none());

    let mut finalized_frame = None;
    while let Some(event) = event_rx.recv().await {
        if let AgentActorEvent::StepFinalized { frame, .. } = event {
            finalized_frame = Some(frame);
            break;
        }
    }

    let frame = finalized_frame.expect("expected StepFinalized event with frame");
    assert_eq!(frame.tools_result.as_ref().map(Vec::len), Some(1));
    assert!(frame.metrics.call_model_duration_ms.is_some());
    assert!(frame.metrics.call_tools_duration_ms.is_some());
}

#[tokio::test]
async fn test_agent_actor_records_error_metrics_when_max_iterations_exceeded() {
    let model = MockModel {
        events: vec![
            StreamEvent::Delta(PartDelta::Text {
                text: "done".to_string(),
            }),
            StreamEvent::Finished {
                reason: Some(FinishReason::Stop),
            },
        ],
    };

    let executor = GenericToolExecutor::new();
    let mut context = Context::new();
    context.add_message(Message::user("hello"));

    let mut actor = AgentActor::new(
        model,
        GenerateRequest::new("mock-model", Vec::new()),
        executor,
        context,
    );
    actor.state_mut().metrics.execution.max_iterations = 0;

    let result = actor.run_step(None).await;

    assert!(matches!(result, crate::agent::StepResult::Error(_)));
    assert_eq!(actor.state().metrics.execution.iteration, 0);
    assert_eq!(
        actor.state().metrics.execution.terminal_reason,
        Some(AgentTerminalReason::Failed)
    );
    assert_eq!(actor.state().metrics.errors.count, 1);
    assert!(
        actor
            .state()
            .metrics
            .errors
            .last_error
            .as_deref()
            .unwrap_or_default()
            .contains("max iter limit 0 exceeded")
    );
}
