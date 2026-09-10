use serde::{Deserialize, Serialize};

/// 用量
///
/// 域类型：`total_tokens` 一定存在，wire 层的可空/别名差异在协议层收敛。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    pub prompt_cache_hit_tokens: Option<u32>,
    pub prompt_cache_miss_tokens: Option<u32>,
}
