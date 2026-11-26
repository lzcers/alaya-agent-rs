use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::agent::{
    CallToolResult,
    agent_actor::lifecycle::{LifeCycleInterrupt, StepFrame},
    tools::ToolCall,
};
use crate::{ContentPart, Message};

/// Agent 执行错误类型
#[derive(Debug, Clone, thiserror::Error, Deserialize, Serialize)]
pub enum AgentError {
    #[error("Cancelled by user")]
    Cancelled,
    #[error("Operation timed out")]
    Timeout,
    #[error("Model response is not expected")]
    ModelRspErr,
    #[error("life cycle error: {0}")]
    Parse(#[from] LifeCycleInterrupt),
}

/// Actor 产生的事件
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum AgentActorEvent {
    /// 流式文本片段
    ContentChunk(String),
    /// 推理内容片段
    ReasoningChunk(String),
    /// 流式音频片段
    AudioChunk {
        data: Option<String>,
        transcript: Option<String>,
        format: Option<String>,
    },
    /// 产出一个完整的内容块（图片、文件、工具调用等）
    PartCompleted(ContentPart),
    /// 模型响应完成，包含完整的多模态消息；此时 step 尚未提交
    StepCompleted { message: Message },
    /// 单步结果已提交，表示最终落地到状态/上下文的结果
    StepFinalized {
        result: StepResult,
        frame: StepFrame,
    },
    /// 模型请求工具调用
    ToolCalls(Vec<ToolCall>),
    /// 单个工具执行完成
    ToolResult {
        call_id: String,
        success: bool,
        output: String,
    },
    /// 一次迭代完成
    Iteration {
        iteration: usize,
        message_count: usize,
    },
    /// 公共 hook 发送的自定义事件
    HookEvent {
        hook: String,
        kind: String,
        payload: Value,
    },
    /// 当前 step 提交后触发的 max-iterations 终态事件
    MaxIterations { iteration: usize },
    /// `run_loop()` 在 `Done` step 完成提交后发送的 actor 级终态事件
    Completed,
    /// 用户取消执行
    Cancelled,
    /// 错误终态事件
    Error(AgentError),
    /// 请求用户输入
    AskUser { question: String, input_id: String },
}

/// 发送给 Actor 的控制命令
#[derive(Debug)]
pub enum AgentActorCommand {
    Pause,
    Continue,
    Cancel,
    UserInput { input: String, input_id: String },
}

/// Actor 的外部控制句柄
pub struct AgentActorHandle {
    pub cmd_tx: mpsc::Sender<AgentActorCommand>,
    pub event_rx: mpsc::Receiver<AgentActorEvent>,
}

impl AgentActorHandle {
    pub async fn pause(&self) {
        let _ = self.cmd_tx.send(AgentActorCommand::Pause).await;
    }

    pub async fn resume(&self) {
        let _ = self.cmd_tx.send(AgentActorCommand::Continue).await;
    }

    pub async fn cancel(&self) {
        let _ = self.cmd_tx.send(AgentActorCommand::Cancel).await;
    }

    pub async fn provide_input(&self, input: String, input_id: String) {
        let _ = self
            .cmd_tx
            .send(AgentActorCommand::UserInput { input, input_id })
            .await;
    }

    pub fn is_finished(&self) -> bool {
        self.event_rx.is_closed()
    }

    pub async fn wait(mut self) -> Vec<AgentActorEvent> {
        let mut events = Vec::new();
        while let Some(event) = self.event_rx.recv().await {
            let is_terminal = matches!(
                event,
                AgentActorEvent::Completed
                    | AgentActorEvent::Cancelled
                    | AgentActorEvent::Error(_)
                    | AgentActorEvent::MaxIterations { .. }
            );
            events.push(event);
            if is_terminal {
                break;
            }
        }
        events
    }
}

/// 单步执行结果
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum StepResult {
    /// 需要继续执行（有工具调用）
    Continue {
        message: Message,
        tools_call: Vec<ToolCall>,
        tools_result: Vec<CallToolResult>,
    },
    /// 执行完成（无工具调用）
    Done { message: Message },
    /// 执行出错
    Error(AgentError),
}
