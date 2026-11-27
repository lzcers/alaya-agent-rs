use crate::{
    AgentActorEvent,
    agent_actor::lifecycle::{LifeCycle, LifeCycleFlow, LifeCycleResult},
    hooks::{HookName, LifeCycleContext, LifeCycleHook},
};
use alaya_model::{PartDelta, StreamEvent};

pub struct SendModelEvtHook;

impl SendModelEvtHook {
    pub fn new() -> Self {
        Self {}
    }
}

#[async_trait::async_trait]
impl LifeCycleHook for SendModelEvtHook {
    fn name(&self) -> HookName {
        "send_model_evt"
    }
    fn priority(&self) -> i32 {
        0
    }
    fn on(&self, stage: &LifeCycle) -> bool {
        matches!(stage, LifeCycle::OnModelEvent | LifeCycle::AfterCallModel)
    }
    async fn handle(&mut self, ctx: &mut LifeCycleContext) -> LifeCycleFlow {
        let Some(tx) = ctx.agent_tx.clone() else {
            return LifeCycleFlow::Continue(LifeCycleResult::None);
        };

        let mapped = match ctx.stage {
            LifeCycle::OnModelEvent => match ctx.model_event.as_ref() {
                Some(StreamEvent::Delta(PartDelta::Text { text })) if !text.is_empty() => {
                    Some(AgentActorEvent::ContentChunk(text.clone()))
                }
                Some(StreamEvent::Delta(PartDelta::Reasoning { text })) if !text.is_empty() => {
                    Some(AgentActorEvent::ReasoningChunk(text.clone()))
                }
                Some(StreamEvent::Delta(PartDelta::Audio {
                    data,
                    transcript,
                    format,
                })) => Some(AgentActorEvent::AudioChunk {
                    data: data.clone(),
                    transcript: transcript.clone(),
                    format: format.clone(),
                }),
                Some(StreamEvent::Part(part)) => Some(AgentActorEvent::PartCompleted(part.clone())),
                _ => None,
            },
            LifeCycle::AfterCallModel => ctx.frame.model_output.as_ref().map(|output| {
                AgentActorEvent::StepCompleted {
                    message: output.message.clone(),
                }
            }),
            _ => None,
        };

        if let Some(event) = mapped {
            let _ = tx.send(event).await;
        }

        LifeCycleFlow::Continue(LifeCycleResult::None)
    }
}
