//! Model-agnostic conversation state, persistence, and compaction.

use serde::{Deserialize, Serialize};

use crate::{Message, Usage};

/// Conversation compaction thresholds.
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
    pub epoch: u64,
    pub messages: Vec<Message>,
    pub usage: CacheUsageTotals,
}

pub struct Conversation {
    messages: Vec<Message>,
    usage: CacheUsageTotals,
    epoch: u64, // 压缩了多少轮
    max_messages: usize,
    max_chars: usize,
}

impl Conversation {
    pub fn new(system_prompt: String, config: ConversationConfig) -> Self {
        Self {
            messages: vec![Message::system(system_prompt)],
            max_messages: config.max_messages,
            max_chars: config.max_chars,
            epoch: 0,
            usage: CacheUsageTotals::default(),
        }
    }

    pub fn snapshot(&self) -> ConversationSnapshot {
        ConversationSnapshot {
            epoch: self.epoch,
            messages: self.messages.clone(),
            usage: self.usage,
        }
    }

    pub fn from_snapshot(
        config: ConversationConfig,
        snapshot: ConversationSnapshot,
    ) -> Result<Self, String> {
        validate_snapshot(&snapshot)?;
        Ok(Self {
            max_messages: config.max_messages,
            max_chars: config.max_chars,
            epoch: snapshot.epoch,
            messages: snapshot.messages,
            usage: snapshot.usage,
        })
    }

    pub fn restore(&mut self, snapshot: ConversationSnapshot) -> Result<(), String> {
        validate_snapshot(&snapshot)?;
        self.epoch = snapshot.epoch;
        self.messages = snapshot.messages;
        self.usage = snapshot.usage;
        Ok(())
    }

    /// Restores conversation history while retaining usage already incurred by the current run.
    pub fn restore_history(&mut self, snapshot: ConversationSnapshot) -> Result<(), String> {
        validate_snapshot(&snapshot)?;
        self.epoch = snapshot.epoch;
        self.messages = snapshot.messages;
        Ok(())
    }

    pub fn reset(&mut self) {
        debug_assert!(matches!(
            self.messages.first(),
            Some(Message::System { .. })
        ));
        self.messages.truncate(1);
        self.epoch = 0;
        self.usage = CacheUsageTotals::default();
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn messages_mut(&mut self) -> &mut Vec<Message> {
        &mut self.messages
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn record_usage(&mut self, usage: Option<Usage>) {
        self.usage.record(usage);
    }

    /// 如果消息数量或字符数超过阈值，执行压缩并返回压缩前的快照。
    ///
    /// 调用方可在后续操作失败时使用 `restore_history` 回滚到压缩前状态。
    pub fn compact_if_needed(&mut self) -> Option<ConversationSnapshot> {
        let chars = self
            .messages
            .iter()
            .map(|message| message.content().chars().count())
            .sum::<usize>();
        if self.messages.len() < self.max_messages && chars < self.max_chars {
            return None;
        }
        let snapshot = ConversationSnapshot {
            epoch: self.epoch,
            messages: self.messages.clone(),
            usage: self.usage,
        };
        let system_message = self
            .messages
            .first()
            .cloned()
            .expect("conversation must start with a system message");
        self.epoch += 1;
        self.messages = vec![
            system_message,
            Message::user(format!(
                "{{\"type\":\"conversation_epoch_checkpoint\",\"epoch\":{}}}",
                self.epoch
            )),
        ];
        Some(snapshot)
    }
}

fn validate_snapshot(snapshot: &ConversationSnapshot) -> Result<(), String> {
    if matches!(snapshot.messages.first(), Some(Message::System { .. })) {
        Ok(())
    } else {
        Err("conversation snapshot must start with a system message".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_preserves_the_snapshot_system_message() {
        let mut conversation = Conversation::new(
            "constructor prompt".to_string(),
            ConversationConfig::default(),
        );
        conversation
            .restore(ConversationSnapshot {
                epoch: 7,
                messages: vec![
                    Message::system("snapshot prompt"),
                    Message::user("previous turn"),
                ],
                usage: CacheUsageTotals {
                    requests: 1,
                    ..CacheUsageTotals::default()
                },
            })
            .unwrap();

        conversation.reset();

        assert_eq!(
            conversation.snapshot(),
            ConversationSnapshot {
                epoch: 0,
                messages: vec![Message::system("snapshot prompt")],
                usage: CacheUsageTotals::default(),
            }
        );
    }

    #[test]
    fn restore_history_preserves_usage() {
        let mut conversation =
            Conversation::new("system".to_string(), ConversationConfig::default());
        let snapshot = conversation.snapshot();
        conversation.push(Message::user("uncommitted"));
        conversation.record_usage(Some(Usage {
            prompt_tokens: 10,
            completion_tokens: 2,
            total_tokens: 12,
            prompt_cache_hit_tokens: Some(4),
            prompt_cache_miss_tokens: Some(6),
        }));

        conversation.restore_history(snapshot).unwrap();

        let restored = conversation.snapshot();
        assert_eq!(restored.messages, vec![Message::system("system")]);
        assert_eq!(restored.usage.requests, 1);
        assert_eq!(restored.usage.prompt_tokens, 10);
    }
}
