use super::AgentActor;
use crate::{Context, ToolExecutor};
use alaya_model::capability::{GenerateCapability, GenerateRequest};

/// Agent Actor 构建器
pub struct AgentActorBuilder<M, E>
where
    M: GenerateCapability + Send + Sync + 'static,
    E: ToolExecutor + Send + 'static,
{
    model: M,
    request: GenerateRequest,
    tool_executor: E,
    context: Context,
    max_iterations: usize,
    user_id: String,
}

impl<M, E> AgentActorBuilder<M, E>
where
    M: GenerateCapability + Send + Sync + 'static,
    E: ToolExecutor + Send + 'static,
{
    pub fn new(model: M, request: GenerateRequest, tool_executor: E) -> Self {
        Self {
            model,
            request,
            tool_executor,
            context: Context::new(),
            max_iterations: 10,
            user_id: "default".to_string(),
        }
    }

    pub fn context(mut self, context: Context) -> Self {
        self.context = context;
        self
    }

    pub fn max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = user_id.into();
        self
    }

    pub fn build(self) -> AgentActor<M, E> {
        let mut actor = AgentActor::with_runtime_hooks(
            self.model,
            self.request,
            self.tool_executor,
            self.context,
        );
        actor.state.metrics.execution.max_iterations = self.max_iterations;
        actor.state.user_id = self.user_id;
        actor
    }
}
