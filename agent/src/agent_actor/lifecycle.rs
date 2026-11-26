use std::ops::ControlFlow;
use std::time::Duration;

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use super::AgentActorEvent;
use crate::hooks::LifeCycleHook;
use crate::hooks::ask_user::AskUserHook;
use crate::hooks::execution_policy::ExecutionPolicyHook;
use crate::hooks::metrics::MetricsHook;
use crate::hooks::send_model_evt::SendModelEvtHook;
use crate::tools::execute_tools;
use crate::{AgentState, ToolCall, ToolExecutor, ToolResult};
use alaya_model::capability::{
    FinishReason, GenerateCapability, GenerateRequest, StreamAccumulator, StreamEvent,
    format_model_error,
};
use alaya_model::{Message, Usage};

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub enum LifeCycle {
    BeforeStep,
    BeforeCallModel,
    OnModelEvent,
    AfterCallModel,
    BeforeCallTools,
    AfterCallTools,
    AfterStep,
}

pub enum LifeCycleResult {
    None,
    Frame(StepFrame),
}

pub enum StepExecutionResult {
    Frame(StepFrame),
    AskUser { question: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum LifeCycleInterrupt {
    HookError(LifeCycle, String, String),
    ToolError(String),
    ModelError(String),
    AskUser { question: String },
}

impl std::fmt::Display for LifeCycleInterrupt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HookError(stage, hook, msg) => {
                write!(f, "hook error: {:?} {} {}", stage, hook, msg)
            }
            Self::ToolError(err) => write!(f, "tool error: {}", err),
            Self::ModelError(err) => write!(f, "model error: {}", err),
            Self::AskUser { question } => write!(f, "ask user: {}", question),
        }
    }
}

impl std::error::Error for LifeCycleInterrupt {}

impl LifeCycleInterrupt {
    pub fn hook_error(stage: &LifeCycle, hook_name: &str, msg: String) -> Self {
        Self::HookError(stage.clone(), hook_name.to_string(), msg)
    }
    pub fn tool_error(err: String) -> Self {
        Self::ToolError(err)
    }
    pub fn model_error(err: String) -> Self {
        Self::ModelError(err)
    }
    pub fn ask_user(question: String) -> Self {
        Self::AskUser { question }
    }
}

pub type LifeCycleFlow = ControlFlow<LifeCycleInterrupt, LifeCycleResult>;

/// 生命周期上下文，给 Hook 使用
pub struct LifeCycleContext {
    pub stage: LifeCycle,
    pub state: AgentState,
    pub frame: StepFrame,
    pub model_event: Option<StreamEvent>,
    pub agent_tx: Option<mpsc::Sender<AgentActorEvent>>,
}

impl LifeCycleContext {
    pub fn set_stage(&mut self, stage: &LifeCycle) {
        self.stage = stage.clone();
    }
    pub fn set_state(&mut self, state: AgentState) {
        self.state = state;
    }
    pub fn set_frame(&mut self, frame: StepFrame) {
        self.frame = frame;
    }
    pub fn set_agent_tx(&mut self, agent_tx: Option<mpsc::Sender<AgentActorEvent>>) {
        self.agent_tx = agent_tx;
    }
    pub fn set_model_event(&mut self, model_event: &StreamEvent) {
        self.model_event = Some(model_event.clone());
    }
    pub fn set_frame_model_output(&mut self, model_output: ModelOutput) {
        self.frame.set_model_output(model_output);
    }
    pub fn set_frame_tools_result(&mut self, tools_result: Vec<ToolResult>) {
        self.frame.set_tools_result(tools_result);
    }
    pub fn set_frame_token_usage(&mut self, token_usage: Usage) {
        self.frame.set_token_usage(token_usage);
    }
    pub fn set_frame_call_model_duration_ms(&mut self, duration_ms: u32) {
        self.frame.set_call_model_duration_ms(duration_ms);
    }
    pub fn set_frame_call_tools_duration_ms(&mut self, duration_ms: u32) {
        self.frame.set_call_tools_duration_ms(duration_ms);
    }
}

impl Default for LifeCycleContext {
    fn default() -> Self {
        Self {
            stage: LifeCycle::BeforeStep,
            state: Default::default(),
            frame: Default::default(),
            model_event: None,
            agent_tx: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelOutput {
    pub message: Message,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<FinishReason>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StepMetrics {
    pub call_model_duration_ms: Option<u32>,
    pub call_tools_duration_ms: Option<u32>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct StepFrame {
    pub model_output: Option<ModelOutput>,
    pub tools_result: Option<Vec<ToolResult>>,
    pub token_usage: Option<Usage>,
    pub metrics: StepMetrics,
}

impl StepFrame {
    pub fn get_tools_call(&self) -> Option<Vec<ToolCall>> {
        let calls: Vec<ToolCall> = self
            .model_output
            .as_ref()?
            .message
            .tool_calls()
            .into_iter()
            .cloned()
            .collect();
        (!calls.is_empty()).then_some(calls)
    }
    pub fn set_model_output(&mut self, model_output: ModelOutput) {
        self.model_output = Some(model_output);
    }
    pub fn set_tools_result(&mut self, tools_result: Vec<ToolResult>) {
        self.tools_result = Some(tools_result);
    }
    pub fn set_token_usage(&mut self, token_usage: Usage) {
        self.token_usage = Some(token_usage);
    }
    pub fn set_call_model_duration_ms(&mut self, duration_ms: u32) {
        self.metrics.call_model_duration_ms = Some(duration_ms);
    }
    pub fn set_call_tools_duration_ms(&mut self, duration_ms: u32) {
        self.metrics.call_tools_duration_ms = Some(duration_ms);
    }
}

pub(super) struct StepLifeCycle {
    hooks: Vec<Box<dyn LifeCycleHook>>,
    ctx: LifeCycleContext,
}

impl StepLifeCycle {
    pub(super) fn new(state: AgentState) -> Self {
        let mut ctx = LifeCycleContext::default();
        ctx.set_state(state);
        Self {
            ctx,
            hooks: vec![
                Box::new(ExecutionPolicyHook::new()),
                Box::new(MetricsHook::new()),
                Box::new(SendModelEvtHook::new()),
                Box::new(AskUserHook::new()),
            ],
        }
    }

    pub(super) async fn start(
        &mut self,
        model: &(dyn GenerateCapability + Sync),
        request: &GenerateRequest,
        tool_executor: &dyn ToolExecutor,
        event_tx: Option<&mpsc::Sender<AgentActorEvent>>,
    ) -> LifeCycleFlow {
        self.ctx.set_agent_tx(event_tx.cloned());
        let messages = self.ctx.state.get_message();
        let tools = tool_executor.tools().clone();
        let tools = (!tools.is_empty()).then_some(tools);

        self.call_life_cycle_hook(LifeCycle::BeforeStep).await?;
        self.call_life_cycle_hook(LifeCycle::BeforeCallModel)
            .await?;

        let request = request.clone().with_messages(messages).with_tools(tools);
        let mut stream = match model.generate_stream(request).await {
            Ok(stream) => stream,
            Err(err) => {
                return Self::break_step(LifeCycleInterrupt::model_error(format_model_error(&err)));
            }
        };

        let mut acc = StreamAccumulator::new();
        while let Some(item) = stream.next().await {
            let event = match item {
                Ok(event) => event,
                Err(err) => {
                    return Self::break_step(LifeCycleInterrupt::model_error(format_model_error(
                        &err,
                    )));
                }
            };
            acc.push(&event);
            self.ctx.set_model_event(&event);
            self.call_life_cycle_hook(LifeCycle::OnModelEvent).await?;
        }

        let response = acc.finish();
        if let Some(usage) = response.usage {
            self.ctx.set_frame_token_usage(usage);
        }
        self.ctx.set_frame_model_output(ModelOutput {
            message: response.message,
            finish_reason: response.finish_reason,
        });

        self.call_life_cycle_hook(LifeCycle::AfterCallModel).await?;

        if let Some(tools_call) = self.ctx.frame.get_tools_call() {
            self.call_life_cycle_hook(LifeCycle::BeforeCallTools)
                .await?;

            match execute_tools(tool_executor, &tools_call, Some(Duration::from_secs(120))).await {
                Ok(results) => self.ctx.set_frame_tools_result(results),
                Err(err) => {
                    return Self::break_step(LifeCycleInterrupt::tool_error(err.to_string()));
                }
            }
            self.call_life_cycle_hook(LifeCycle::AfterCallTools).await?;
        }

        self.call_life_cycle_hook(LifeCycle::AfterStep).await?;

        Self::continue_step_with_result(LifeCycleResult::Frame(self.ctx.frame.clone()))
    }

    fn break_step(err: LifeCycleInterrupt) -> LifeCycleFlow {
        LifeCycleFlow::Break(err)
    }

    fn continue_step() -> LifeCycleFlow {
        LifeCycleFlow::Continue(LifeCycleResult::None)
    }
    fn continue_step_with_result(result: LifeCycleResult) -> LifeCycleFlow {
        LifeCycleFlow::Continue(result)
    }

    async fn call_life_cycle_hook(&mut self, lifecycle: LifeCycle) -> LifeCycleFlow {
        self.ctx.set_stage(&lifecycle);
        for hook in &mut self.hooks {
            if hook.on(&lifecycle) {
                let result = hook.handle(&mut self.ctx).await;
                if let LifeCycleFlow::Break(_) = result {
                    return result;
                }
            }
        }
        Self::continue_step()
    }
}
