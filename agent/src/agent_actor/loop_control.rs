use std::sync::Arc;

use tokio::sync::mpsc;

use super::lifecycle::{LifeCycleFlow, LifeCycleInterrupt, LifeCycleResult, StepLifeCycle};
use super::{AgentActor, AgentActorCommand, AgentActorEvent, AgentActorHandle, StepResult};
use crate::agent_actor::lifecycle::StepFrame;
use crate::{AgentError, AgentTerminalReason, JobState, ToolExecutor};
use alaya_model::capability::GenerateCapability;
use alaya_model::{ContentPart, Message};

#[derive(Clone)]
pub enum LoopState {
    Runnable,
    Paused,
    WaitingForUserInput(String),
    Finished(Finalization),
}

#[derive(Clone)]
pub enum Finalization {
    Emit(AgentTerminalReason),
    Silent,
}

impl<M, E> AgentActor<M, E>
where
    M: GenerateCapability + Send + Sync,
    E: ToolExecutor + Send,
{
    pub async fn run_step(
        &mut self,
        event_tx: Option<mpsc::Sender<AgentActorEvent>>,
    ) -> StepResult {
        self.state.state = JobState::Running;
        self.state.metrics.mark_started();
        let model = Arc::clone(&self.model);
        let tool_executor = Arc::clone(&self.tool_executor);
        let mut lifecycle = StepLifeCycle::new(self.state.clone());

        let lifecycle_flow = lifecycle
            .start(
                model.as_ref(),
                &self.model_name,
                &self.generate_config,
                tool_executor.as_ref(),
                event_tx.as_ref(),
            )
            .await;

        match lifecycle_flow {
            LifeCycleFlow::Continue(LifeCycleResult::Frame(frame)) => {
                self.apply_step_metrics(&frame);
                let result = Self::step_result_from_frame(&frame);
                self.apply_step_result(&result);
                self.emit_step_result_events(event_tx.as_ref(), &frame, &result)
                    .await;
                result
            }
            LifeCycleFlow::Break(interrupt) => match interrupt {
                LifeCycleInterrupt::AskUser { question } => {
                    self.ask_user(question, event_tx).await;
                    StepResult::Continue {
                        message: Message::assistant(""),
                        tools_call: Vec::new(),
                        tools_result: Vec::new(),
                    }
                }
                _ => {
                    let err = AgentError::Parse(interrupt);
                    self.apply_step_error(&err);
                    self.emit_error_events(event_tx.as_ref(), &err).await;
                    StepResult::Error(err)
                }
            },
            _ => {
                let err = AgentError::Parse(LifeCycleInterrupt::model_error(
                    "unexpected lifecycle result".to_string(),
                ));
                self.apply_step_error(&err);
                self.emit_error_events(event_tx.as_ref(), &err).await;
                StepResult::Error(err)
            }
        }
    }

    pub fn run_loop(mut self) -> AgentActorHandle {
        let (cmd_tx, mut cmd_rx) = mpsc::channel::<AgentActorCommand>(16);
        let (event_tx, event_rx) = mpsc::channel::<AgentActorEvent>(64);

        tokio::spawn(async move {
            let mut loop_state = LoopState::Runnable;

            loop {
                if event_tx.is_closed() {
                    break;
                }

                loop_state = self.drain_commands(loop_state, &mut cmd_rx);

                let current_state = std::mem::replace(&mut loop_state, LoopState::Runnable);
                match current_state {
                    LoopState::Runnable => {
                        let result = self.run_step(Some(event_tx.clone())).await;
                        loop_state = self.transition_step_result(result);
                    }
                    LoopState::Paused => {
                        loop_state = self.wait_while_paused(&mut cmd_rx, &event_tx).await;
                    }
                    LoopState::WaitingForUserInput(_) => {
                        loop_state = self.wait_for_user_input(&mut cmd_rx, &event_tx).await;
                    }
                    LoopState::Finished(finalization) => {
                        self.emit_terminal_event(&event_tx, &finalization).await;
                        break;
                    }
                };

                if let LoopState::Finished(finalization) = &loop_state {
                    self.emit_terminal_event(&event_tx, finalization).await;
                    break;
                }
            }
        });

        AgentActorHandle { cmd_tx, event_rx }
    }

    fn drain_commands(
        &mut self,
        mut loop_state: LoopState,
        cmd_rx: &mut mpsc::Receiver<AgentActorCommand>,
    ) -> LoopState {
        while let Ok(cmd) = cmd_rx.try_recv() {
            loop_state = self.transition_command(loop_state, cmd);
        }
        loop_state
    }

    async fn wait_while_paused(
        &mut self,
        cmd_rx: &mut mpsc::Receiver<AgentActorCommand>,
        event_tx: &mpsc::Sender<AgentActorEvent>,
    ) -> LoopState {
        tokio::select! {
            Some(cmd) = cmd_rx.recv() => {
                self.transition_command(LoopState::Paused, cmd)
            }
            _ = event_tx.closed() => {
                self.state.state = JobState::Cancelled;
                self.state.metrics.mark_finished(AgentTerminalReason::Cancelled);
                LoopState::Finished(Finalization::Emit(AgentTerminalReason::Cancelled))
            }
        }
    }

    async fn wait_for_user_input(
        &mut self,
        cmd_rx: &mut mpsc::Receiver<AgentActorCommand>,
        event_tx: &mpsc::Sender<AgentActorEvent>,
    ) -> LoopState {
        if let Some((input_id, _)) = &self.pending_user_input {
            tokio::select! {
                Some(cmd) = cmd_rx.recv() => {
                    self.transition_command(LoopState::WaitingForUserInput(input_id.clone()), cmd)
                }
                _ = event_tx.closed() => {
                    self.state.state = JobState::Cancelled;
                    self.state.metrics.mark_finished(AgentTerminalReason::Cancelled);
                    self.pending_user_input = None;
                    LoopState::Finished(Finalization::Emit(AgentTerminalReason::Cancelled))
                }
            }
        } else {
            LoopState::Runnable
        }
    }

    fn transition_command(&mut self, loop_state: LoopState, cmd: AgentActorCommand) -> LoopState {
        match loop_state {
            LoopState::Runnable => match cmd {
                AgentActorCommand::Pause => {
                    self.state.state = JobState::Paused;
                    self.state.metrics.mark_active();
                    LoopState::Paused
                }
                AgentActorCommand::Continue => {
                    self.state.metrics.mark_active();
                    LoopState::Runnable
                }
                AgentActorCommand::Cancel => {
                    self.state.state = JobState::Cancelled;
                    self.state
                        .metrics
                        .mark_finished(AgentTerminalReason::Cancelled);
                    LoopState::Finished(Finalization::Emit(AgentTerminalReason::Cancelled))
                }
                AgentActorCommand::UserInput { .. } => LoopState::Runnable,
            },
            LoopState::Paused => match cmd {
                AgentActorCommand::Pause => LoopState::Paused,
                AgentActorCommand::Continue => {
                    self.state.state = JobState::Running;
                    self.state.metrics.mark_active();
                    LoopState::Runnable
                }
                AgentActorCommand::Cancel => {
                    self.state.state = JobState::Cancelled;
                    self.state
                        .metrics
                        .mark_finished(AgentTerminalReason::Cancelled);
                    LoopState::Finished(Finalization::Emit(AgentTerminalReason::Cancelled))
                }
                AgentActorCommand::UserInput { .. } => LoopState::Paused,
            },
            LoopState::WaitingForUserInput(expected_input_id) => match cmd {
                AgentActorCommand::UserInput { input, input_id } => {
                    if input_id == expected_input_id {
                        self.state.context.add_message(Message::user(input));
                        self.state.state = JobState::Running;
                        self.pending_user_input = None;
                        LoopState::Runnable
                    } else {
                        LoopState::WaitingForUserInput(expected_input_id)
                    }
                }
                AgentActorCommand::Cancel => {
                    self.state.state = JobState::Cancelled;
                    self.state
                        .metrics
                        .mark_finished(AgentTerminalReason::Cancelled);
                    self.pending_user_input = None;
                    LoopState::Finished(Finalization::Emit(AgentTerminalReason::Cancelled))
                }
                _ => LoopState::WaitingForUserInput(expected_input_id),
            },
            LoopState::Finished(finalization) => LoopState::Finished(finalization),
        }
    }

    fn transition_step_result(&self, result: StepResult) -> LoopState {
        match result {
            StepResult::Continue { tools_call, .. } => {
                if tools_call.iter().any(|tool| tool.name == "ask_user") {
                    if self.has_pending_user_input() {
                        if let Some((input_id, _)) = &self.pending_user_input {
                            LoopState::WaitingForUserInput(input_id.clone())
                        } else {
                            LoopState::Runnable
                        }
                    } else {
                        LoopState::Runnable
                    }
                } else {
                    LoopState::Runnable
                }
            }
            StepResult::Done { .. } => {
                LoopState::Finished(Finalization::Emit(AgentTerminalReason::Completed))
            }
            StepResult::Error(_) => LoopState::Finished(Finalization::Silent),
        }
    }

    async fn emit_terminal_event(
        &self,
        event_tx: &mpsc::Sender<AgentActorEvent>,
        finalization: &Finalization,
    ) {
        match finalization {
            Finalization::Emit(reason) => match reason {
                AgentTerminalReason::Completed => {
                    Self::send_event(Some(event_tx), AgentActorEvent::Completed).await;
                }
                AgentTerminalReason::Cancelled => {
                    Self::send_event(Some(event_tx), AgentActorEvent::Cancelled).await;
                }
                AgentTerminalReason::Failed => {}
            },
            Finalization::Silent => {}
        }
    }

    fn apply_step_result(&mut self, result: &StepResult) {
        match result {
            StepResult::Continue {
                message,
                tools_result,
                ..
            } => {
                self.state.metrics.increment_iteration();
                self.state.state = JobState::Running;
                self.state.context.add_message(message.clone());

                for tool_result in tools_result {
                    self.state.context.add_message(Message::tool_result_parts(
                        tool_result.call_id.clone(),
                        vec![ContentPart::text(tool_result.output_text())],
                        !tool_result.success,
                    ));
                }
            }
            StepResult::Done { message } => {
                self.state.metrics.increment_iteration();
                self.state.state = JobState::Completed;
                self.state.context.add_message(message.clone());
                self.state
                    .metrics
                    .mark_finished(AgentTerminalReason::Completed);
            }
            StepResult::Error(err) => self.apply_step_error(err),
        }
    }

    fn apply_step_metrics(&mut self, frame: &StepFrame) {
        self.state.metrics.mark_active();
        if let Some(usage) = frame.token_usage.as_ref() {
            self.state.metrics.add_usage(usage);
        }
        if let Some(duration_ms) = frame.metrics.call_model_duration_ms {
            self.state.metrics.add_model_duration(duration_ms);
        }
        if let Some(duration_ms) = frame.metrics.call_tools_duration_ms {
            self.state.metrics.add_tool_duration(duration_ms);
        }
        if let Some(results) = frame.tools_result.as_ref() {
            let success_count = results.iter().filter(|result| result.success).count();
            let failure_count = results.len().saturating_sub(success_count);
            self.state
                .metrics
                .add_tool_results(success_count, failure_count);
        }
    }

    fn apply_step_error(&mut self, err: &AgentError) {
        match err {
            AgentError::Cancelled => {
                self.state.state = JobState::Cancelled;
                self.state
                    .metrics
                    .mark_finished(AgentTerminalReason::Cancelled);
            }
            _ => {
                self.state.state = JobState::Failed;
                self.state.metrics.record_error(err.to_string());
                self.state
                    .metrics
                    .mark_finished(AgentTerminalReason::Failed);
            }
        }
    }

    fn step_result_from_frame(frame: &StepFrame) -> StepResult {
        match frame.model_output.as_ref() {
            Some(model_output) => match frame.get_tools_call() {
                Some(tools_call) => StepResult::Continue {
                    message: model_output.message.clone(),
                    tools_call,
                    tools_result: frame.tools_result.clone().unwrap_or_default(),
                },
                None => StepResult::Done {
                    message: model_output.message.clone(),
                },
            },
            None => StepResult::Error(AgentError::ModelRspErr),
        }
    }

    async fn emit_step_result_events(
        &self,
        event_tx: Option<&mpsc::Sender<AgentActorEvent>>,
        frame: &StepFrame,
        result: &StepResult,
    ) {
        Self::send_event(
            event_tx,
            AgentActorEvent::StepFinalized {
                result: result.clone(),
                frame: frame.clone(),
            },
        )
        .await;

        match result {
            StepResult::Continue { .. } | StepResult::Done { .. } => {
                Self::send_event(
                    event_tx,
                    AgentActorEvent::Iteration {
                        iteration: self.state.metrics.execution.iteration,
                        message_count: self.state.context.conversation().len(),
                    },
                )
                .await;
            }
            StepResult::Error(AgentError::Cancelled) => {
                Self::send_event(event_tx, AgentActorEvent::Cancelled).await;
            }
            StepResult::Error(err) => {
                Self::send_event(event_tx, AgentActorEvent::Error(err.clone())).await;
            }
        }
    }

    async fn emit_error_events(
        &self,
        event_tx: Option<&mpsc::Sender<AgentActorEvent>>,
        err: &AgentError,
    ) {
        match err {
            AgentError::Cancelled => {
                Self::send_event(event_tx, AgentActorEvent::Cancelled).await;
            }
            _ => {
                Self::send_event(event_tx, AgentActorEvent::Error(err.clone())).await;
            }
        }
    }
}
