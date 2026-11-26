/// 构建器相关实现位于 `builder` 子模块。
mod builder;
/// 单步执行生命周期位于 `lifecycle` 子模块。
pub mod lifecycle;
/// 后台循环控制逻辑位于 `loop_control` 子模块。
mod loop_control;
/// 错误、事件、命令和结果类型位于 `types` 子模块。
mod types;

use std::sync::Arc;

use tokio::sync::mpsc;
use uuid::Uuid;

use crate::{AgentState, Context, Metrics, ToolExecutor};
use alaya_model::capability::{GenerateCapability, GenerateRequest};

pub use builder::AgentActorBuilder;
pub use loop_control::LoopState;
pub use types::{AgentActorCommand, AgentActorEvent, AgentActorHandle, AgentError, StepResult};

/// Agent Actor 负责组合统一多模态模型、工具执行器和运行配置。
pub struct AgentActor<M, E>
where
    M: GenerateCapability + Send + Sync + 'static,
    E: ToolExecutor + Send + 'static,
{
    /// Agent 状态（控制面）
    state: AgentState,
    /// 多模态生成模型
    model: Arc<M>,
    /// 每次 step 使用的生成请求模板；messages 和 tools 会在调用前刷新。
    request: GenerateRequest,
    /// 工具执行器
    tool_executor: Arc<E>,
    /// 等待中的用户输入
    pending_user_input: Option<(String, mpsc::Sender<AgentActorEvent>)>,
}

impl<M, E> AgentActor<M, E>
where
    M: GenerateCapability + Send + Sync,
    E: ToolExecutor + Send,
{
    /// 创建新的 Agent Actor
    pub fn new(model: M, request: GenerateRequest, tool_executor: E, context: Context) -> Self {
        Self::with_runtime_hooks(model, request, tool_executor, context)
    }

    pub(crate) fn with_runtime_hooks(
        model: M,
        request: GenerateRequest,
        tool_executor: E,
        context: Context,
    ) -> Self {
        let default_max_iterations = 10;
        Self {
            state: AgentState {
                job_id: Uuid::new_v4(),
                user_id: "default".to_string(),
                conversation_id: None,
                title: String::new(),
                description: String::new(),
                category: None,
                state: crate::JobState::Pending,
                context,
                metrics: Metrics::with_max_iterations(default_max_iterations),
            },
            model: Arc::new(model),
            request,
            tool_executor: Arc::new(tool_executor),
            pending_user_input: None,
        }
    }

    pub fn state(&self) -> &AgentState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut AgentState {
        &mut self.state
    }

    pub fn context(&self) -> &Context {
        &self.state.context
    }

    pub fn context_mut(&mut self) -> &mut Context {
        &mut self.state.context
    }

    async fn send_event(event_tx: Option<&mpsc::Sender<AgentActorEvent>>, event: AgentActorEvent) {
        if let Some(tx) = event_tx {
            let _ = tx.send(event).await;
        }
    }

    pub async fn ask_user(
        &mut self,
        question: String,
        event_tx: Option<mpsc::Sender<AgentActorEvent>>,
    ) -> LoopState {
        let input_id = Uuid::new_v4().to_string();

        if let Some(tx) = &event_tx {
            Self::send_event(
                Some(tx),
                AgentActorEvent::AskUser {
                    question,
                    input_id: input_id.clone(),
                },
            )
            .await;
        }

        self.state.state = crate::JobState::Paused;
        self.pending_user_input = Some((input_id.clone(), event_tx.unwrap()));
        LoopState::WaitingForUserInput(input_id)
    }

    pub fn has_pending_user_input(&self) -> bool {
        self.pending_user_input.is_some()
    }
}
