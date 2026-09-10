use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub mod ask_user;
pub mod playwright_cli;
pub mod registry;

pub use playwright_cli::PlaywrightCliTool;
pub use registry::{GenericToolExecutor, Tool, ToolRegistry};
use thiserror::Error;

// 模型面向的工具类型属于消息域，这里重导出以保持 `agent::tools::ToolCall` 等路径可用。
pub use crate::message::tool::{ToolCall, ToolDef};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    /// 与 ToolCall 相同的 id，用于关联调用和结果。
    pub id: String,
    /// 工具执行是否成功。
    pub success: bool,
    /// 工具执行的输出，可以是任意 JSON 值。
    /// 如果失败，可以包含错误信息。
    pub output: Value,
}

#[derive(Debug, Error, Clone)]
pub enum ToolExecutorError {
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Execution error: {0}")]
    ExecutionError(String),
}

// 工具执行器独立
#[async_trait]
pub trait ToolExecutor: Sync {
    async fn execute(&self, call: &ToolCall) -> Result<ToolResult, ToolExecutorError>;
    fn tools(&self) -> &Vec<ToolDef>;
}
