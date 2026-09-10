use async_trait::async_trait;
use futures::stream::BoxStream;

use crate::Usage;
use crate::agent::{ToolCall, ToolDef};
use crate::capability::ModelError;
use crate::message::Message;

/// 与厂商无关的推理强度配置。
///
/// 域层只表达「要多少推理」；具体编码成 `reasoning_effort` 还是
/// `thinking: {type: enabled}` 由协议层决定。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reasoning {
    /// OpenAI 风格：`low` / `medium` / `high`
    pub effort: Option<String>,
    /// DeepSeek 风格：显式开关思考模式
    pub thinking: Option<bool>,
}

/// 一次 chat 调用的域请求。
///
/// 这里没有任何传输细节：没有 `stream` 开关，没有厂商私有字段袋。
#[derive(Debug, Clone)]
pub struct ChatRequest {
    /// 模型名，同时是分发层的路由键，也是 wire body 的必填字段。
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// 可用工具定义，协议层负责翻译成 wire 格式。
    pub tools: Option<Vec<ToolDef>>,
    /// 要求模型返回 JSON object。
    pub response_format_json: bool,
    /// 要求流式响应携带 token 用量。
    pub include_usage: bool,
    pub reasoning: Reasoning,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            temperature: None,
            max_tokens: None,
            tools: None,
            response_format_json: false,
            include_usage: false,
            reasoning: Reasoning::default(),
        }
    }

    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_tools(mut self, tools: Option<Vec<ToolDef>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_response_format_json(mut self) -> Self {
        self.response_format_json = true;
        self
    }

    pub fn with_stream_usage(mut self, include_usage: bool) -> Self {
        self.include_usage = include_usage;
        self
    }

    pub fn with_reasoning_effort(mut self, reasoning_effort: impl Into<String>) -> Self {
        self.reasoning.effort = Some(reasoning_effort.into());
        self
    }

    pub fn with_thinking(mut self, enabled: bool) -> Self {
        self.reasoning.thinking = Some(enabled);
        self
    }
}

/// 流式响应的一个增量块。
#[derive(Debug, Clone)]
pub struct ChatChunk {
    pub content: String,
    pub reasoning_content: String,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub usage: Option<Usage>,
}

/// Chat 能力。
///
/// 由协议适配器实现（wire ↔ 域映射），也可由 [`crate::router::ModelRouter`]
/// 实现（按模型名分发到已注册的实现）。
#[async_trait]
pub trait ChatCapability: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<Message, ModelError>;

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<BoxStream<'static, ChatChunk>, ModelError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_builders_only_touch_domain_fields() {
        let request = ChatRequest::new("model", Vec::new())
            .with_response_format_json()
            .with_reasoning_effort("high")
            .with_thinking(true)
            .with_temperature(0.2)
            .with_max_tokens(512)
            .with_stream_usage(true);

        assert!(request.response_format_json);
        assert!(request.include_usage);
        assert_eq!(request.temperature, Some(0.2));
        assert_eq!(request.max_tokens, Some(512));
        assert_eq!(request.reasoning.effort.as_deref(), Some("high"));
        assert_eq!(request.reasoning.thinking, Some(true));
    }

    #[test]
    fn tools_stay_domain_typed() {
        let tool = ToolDef {
            name: "read_file".to_string(),
            description: "read".to_string(),
            parameters: json!({ "type": "object" }),
        };
        let request = ChatRequest::new("model", Vec::new()).with_tools(Some(vec![tool]));

        assert_eq!(request.tools.unwrap()[0].name, "read_file");
    }
}
