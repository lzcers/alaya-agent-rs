//! 工具使用的**模型面向**类型。
//!
//! 这些是模型看到的东西——告诉它有哪些工具可用（[`ToolDef`]）、它要求调用哪个工具
//! （[`ToolCall`]）。它们属于 chat 协议负载，因此住在消息域。
//!
//! 这两个类型都是**纯域形状**：不含任何厂商私有字段。OpenAI 的嵌套
//! `function.arguments` 字符串、流式增量里的 `index`、`type: "function"` 等
//! 形状差异由协议层负责翻译（见 [`crate::protocols::openai::wire::WireToolCall`]）。
//!
//! 工具**执行**侧的类型（`Tool`、`ToolRegistry`、`GenericToolExecutor`、
//! `ToolResult`）留在 [`crate::agent::tools`]——那才是编排层的职责。

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 工具定义，用于告知模型可用的工具。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    /// 工具名称，模型在调用时使用此名称。
    pub name: String,
    /// 工具描述，帮助模型理解何时使用该工具。
    pub description: String,
    /// 参数 JSON Schema，描述工具接受的参数格式。
    /// 通常是一个符合 JSON Schema 规范的对象。
    pub parameters: Value,
}

/// 模型要求调用一个工具。
///
/// 只有三个字段：哪个调用（`id`）、调什么（`name`）、传什么参（`arguments`）。
/// 流式传输中的字段碎片由协议层合并成完整项后再交付给上层。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCall {
    /// 工具调用的唯一标识符，用于将结果与调用关联。
    /// 通常由模型生成，执行结果需原样返回。
    pub id: String,
    /// 工具名称。
    pub name: String,
    /// 已解析的参数。模型给出的参数无法解析为 JSON 时为 [`Value::Null`]，
    /// 协议层会记录告警。
    pub arguments: Value,
}
