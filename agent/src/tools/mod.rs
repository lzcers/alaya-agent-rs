use std::time::Duration;

use async_trait::async_trait;
use futures::future::join_all;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

pub mod ask_user;
pub mod playwright_cli;
pub mod registry;

pub use playwright_cli::PlaywrightCliTool;
pub use registry::{GenericToolExecutor, Tool, ToolRegistry};

// 模型面向的工具类型属于 alaya-model 消息域，这里重导出便于工具执行侧使用。
pub use alaya_model::{ToolCall, ToolDef};

/// 一次工具调用的执行结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// 与 [`ToolCall::id`] 相同的唯一标识符，用于将结果与调用关联。
    pub call_id: String,
    /// 工具名称。
    pub tool_name: String,
    /// 工具执行是否成功。
    pub success: bool,
    /// 工具执行的输出（JSON 值；失败时包含错误信息）。
    pub output: Value,
}

impl ToolResult {
    /// 序列化为适合喂回模型的 JSON 文本。
    pub fn output_text(&self) -> String {
        serde_json::to_string(&self.output).unwrap_or_else(|_| "{}".to_string())
    }
}

#[derive(Debug, Error, Clone)]
pub enum ToolExecutorError {
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
    #[error("Tool execution timed out")]
    Timeout,
}

/// 工具执行器。
#[async_trait]
pub trait ToolExecutor: Sync {
    async fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolExecutorError>;
    fn tools(&self) -> &Vec<ToolDef>;
}

/// 执行单个工具，将失败收敛为 `success: false` 的 [`ToolResult`]。
pub async fn execute_tool(tool_executor: &dyn ToolExecutor, call: &ToolCall) -> ToolResult {
    match tool_executor.execute(call).await {
        Ok(result) if result.success => result,
        Ok(result) => ToolResult {
            call_id: call.id.clone(),
            tool_name: call.name.clone(),
            success: false,
            output: json!({ "error": result.output }),
        },
        Err(err) => ToolResult {
            call_id: call.id.clone(),
            tool_name: call.name.clone(),
            success: false,
            output: json!({ "error": err.to_string() }),
        },
    }
}

/// 并行执行多个工具调用，支持可选超时控制。
pub async fn execute_tools(
    tool_executor: &dyn ToolExecutor,
    calls: &[ToolCall],
    timeout: Option<Duration>,
) -> Result<Vec<ToolResult>, ToolExecutorError> {
    let run = join_all(calls.iter().map(|call| execute_tool(tool_executor, call)));
    match timeout {
        Some(dur) => tokio::time::timeout(dur, run)
            .await
            .map_err(|_| ToolExecutorError::Timeout),
        None => Ok(run.await),
    }
}
