//! Model-agnostic conversation state for one-shot agent calls.
//!
//! 每次 agent 调用构造一个 Conversation（以 system 开头），推入当轮 user 后交模型补全，
//! 再把 assistant 拼回并累计当轮 usage；快照 {epoch, usage, messages} 落库作为审计记录。
//! 压缩机制已移除：跨轮历史由各 agent 的结构化上下文注入，conversation 不再跨轮累积，
//! epoch 恒为 0（字段保留以兼容已持久化的快照）。

use serde::{Deserialize, Serialize};

use crate::{Message, Usage};

/// Conversation 构造配置。
///
/// max_messages / max_chars 原为历史压缩阈值；压缩机制已移除，阈值不再生效，
/// 保留该配置仅为兼容现有构造调用与持久化配置。
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "snake_case")]
pub struct ConversationConfig {
    pub max_messages: usize,
    pub max_chars: usize,
}

impl Default for ConversationConfig {
    fn default() -> Self {
        Self {
            max_messages: 96,
            max_chars: 512_000,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, rename_all = "snake_case")]
pub struct CacheUsageTotals {
    pub requests: u64,
    pub requests_with_usage: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub prompt_cache_hit_tokens: u64,
    pub prompt_cache_miss_tokens: u64,
}

impl CacheUsageTotals {
    fn record(&mut self, usage: Option<Usage>) {
        self.requests += 1;
        let Some(usage) = usage else {
            return;
        };
        self.requests_with_usage += 1;
        self.prompt_tokens += u64::from(usage.prompt_tokens);
        self.completion_tokens += u64::from(usage.completion_tokens);
        self.prompt_cache_hit_tokens +=
            u64::from(usage.prompt_cache_hit_tokens.unwrap_or_default());
        self.prompt_cache_miss_tokens +=
            u64::from(usage.prompt_cache_miss_tokens.unwrap_or_default());
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct ConversationSnapshot {
    /// 压缩轮数；压缩机制已移除，恒为 0，保留字段以兼容已持久化的快照。
    pub epoch: u64,
    pub messages: Vec<Message>,
    pub usage: CacheUsageTotals,
}

/// 一次 agent 调用的消息集合与用量累计。
///
/// 仅承载单次调用：以 system 开头，推入当轮 user，模型返回后拼回 assistant 并记录
/// usage。不再跨调用累积（历史由各 agent 的结构化上下文注入）。
pub struct Conversation {
    messages: Vec<Message>,
    usage: CacheUsageTotals,
    epoch: u64,
}

impl Conversation {
    /// 以 system 提示词开头构造一次全新调用。
    ///
    /// config 的阈值字段在压缩移除后不再生效，仅为兼容构造签名保留。
    pub fn new(system_prompt: String, _config: ConversationConfig) -> Self {
        Self {
            messages: vec![Message::system(system_prompt)],
            usage: CacheUsageTotals::default(),
            epoch: 0,
        }
    }

    pub fn snapshot(&self) -> ConversationSnapshot {
        ConversationSnapshot {
            epoch: self.epoch,
            messages: self.messages.clone(),
            usage: self.usage,
        }
    }

    /// 截断消息到 system 并清零用量与压缩轮次。
    ///
    /// 用于 fate_weaver：每次 advance 从当轮上下文开始，不跨 commit 累积。
    pub fn reset(&mut self) {
        debug_assert!(matches!(
            self.messages.first(),
            Some(Message::System { .. })
        ));
        self.messages.truncate(1);
        self.epoch = 0;
        self.usage = CacheUsageTotals::default();
    }

    /// 仅截断消息到 system，但保留 usage 累计（不同于 Self::reset）。
    ///
    /// 用于无状态重试：每次重试只注入新的 user 输入（feedback 已并入其中），模型不再
    /// 看到先前失败的输出；快照 messages 只含最终一次交换，usage 仍覆盖本轮全部尝试。
    pub fn reset_messages(&mut self) {
        debug_assert!(matches!(
            self.messages.first(),
            Some(Message::System { .. })
        ));
        self.messages.truncate(1);
        self.epoch = 0;
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn record_usage(&mut self, usage: Option<Usage>) {
        self.usage.record(usage);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_shot_call_records_a_single_exchange_and_usage() {
        let mut conversation =
            Conversation::new("system".to_string(), ConversationConfig::default());
        conversation.push(Message::user("turn"));
        conversation.push(Message::assistant("answer"));
        conversation.record_usage(Some(Usage {
            prompt_tokens: 10,
            completion_tokens: 2,
            total_tokens: 12,
            prompt_cache_hit_tokens: Some(4),
            prompt_cache_miss_tokens: Some(6),
        }));

        let snapshot = conversation.snapshot();
        assert_eq!(snapshot.epoch, 0);
        assert_eq!(
            snapshot.messages,
            vec![
                Message::system("system"),
                Message::user("turn"),
                Message::assistant("answer"),
            ]
        );
        assert_eq!(snapshot.usage.requests, 1);
        assert_eq!(snapshot.usage.prompt_tokens, 10);
    }

    #[test]
    fn reset_preserves_the_system_message() {
        let mut conversation = Conversation::new(
            "constructor prompt".to_string(),
            ConversationConfig::default(),
        );
        conversation.push(Message::user("previous turn"));

        conversation.reset();

        assert_eq!(
            conversation.snapshot().messages,
            vec![Message::system("constructor prompt")]
        );
        assert_eq!(conversation.snapshot().usage, CacheUsageTotals::default());
    }

    #[test]
    fn reset_messages_truncates_to_system_but_keeps_usage() {
        let mut conversation =
            Conversation::new("system".to_string(), ConversationConfig::default());
        conversation.push(Message::user("attempt one"));
        conversation.push(Message::assistant("failed output"));
        conversation.record_usage(Some(Usage {
            prompt_tokens: 10,
            completion_tokens: 2,
            total_tokens: 12,
            prompt_cache_hit_tokens: Some(4),
            prompt_cache_miss_tokens: Some(6),
        }));

        conversation.reset_messages();

        let snapshot = conversation.snapshot();
        assert_eq!(snapshot.messages, vec![Message::system("system")]);
        assert_eq!(snapshot.epoch, 0);
        assert_eq!(snapshot.usage.requests, 1);
        assert_eq!(snapshot.usage.prompt_tokens, 10);
        assert_eq!(snapshot.usage.completion_tokens, 2);
    }
}
