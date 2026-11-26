use serde::{Deserialize, Serialize};

/// 模型调用的 Token 用量（厂商无关域类型）。
///
/// 各厂商命名差异（OpenAI `prompt_tokens`/`cached_tokens`、DeepSeek `prompt_cache_hit_tokens`、
/// Anthropic `input_tokens`/`cache_read_input_tokens`、Gemini `promptTokenCount`/`thoughtsTokenCount`）
/// 统一在协议层收敛到此结构。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
}
