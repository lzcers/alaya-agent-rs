use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

use crate::memory::MemoryError;
use crate::{Layer, LayerKind};
use alaya_model::capability::{GenerateCapability, GenerateRequest, ModelError};
use alaya_model::{Message, MessageRole};

#[async_trait]
pub trait SummaryModel: Send + Sync {
    async fn summarize(&self, prompt: &str) -> Result<String, ModelError>;
}

const DEFAULT_SUMMARY_LAYER_NAME: &str = "conversation_summary";
const DEFAULT_SUMMARY_PRIORITY: i32 = 25;
const DEFAULT_SUMMARY_SYSTEM_PROMPT: &str = "You compress layered agent context. Return only a concise summary that will help continue the task later.";

#[derive(Debug, Error)]
pub enum CompressionError {
    #[error("conversation layer `{0}` must contain a JSON array of messages")]
    InvalidConversationLayer(String),
    #[error("layer action `{action}` is incompatible with layer `{layer}`")]
    IncompatibleLayerAction { layer: String, action: String },
    #[error("summary model returned an empty summary")]
    EmptySummary,
    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("memory error: {0}")]
    Memory(#[from] MemoryError),
    #[error("model error: {0}")]
    Model(#[from] ModelError),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuleCompression {
    #[serde(default)]
    pub layer_rules: Vec<LayerRule>,
    #[serde(default)]
    pub conversation: Option<ConversationRule>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRule {
    pub selector: LayerSelector,
    pub action: LayerAction,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LayerSelector {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub kind: Option<LayerKind>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub include_readonly: bool,
}

impl LayerSelector {
    pub(super) fn matches(&self, layer: &Layer) -> bool {
        if layer.meta.readonly && !self.include_readonly {
            return false;
        }

        if let Some(name) = &self.name
            && layer.name != *name
        {
            return false;
        }

        if let Some(kind) = &self.kind
            && layer.kind != *kind
        {
            return false;
        }

        self.tags
            .iter()
            .all(|tag| layer.meta.tags.iter().any(|candidate| candidate == tag))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LayerAction {
    Drop,
    Clear,
    TrimArray { keep_head: usize, keep_tail: usize },
    Replace { value: Value },
}

impl LayerAction {
    pub(super) fn label(&self) -> &'static str {
        match self {
            Self::Drop => "drop",
            Self::Clear => "clear",
            Self::TrimArray { .. } => "trim_array",
            Self::Replace { .. } => "replace",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRule {
    #[serde(default = "default_keep_recent_messages")]
    pub keep_recent_messages: usize,
    #[serde(default = "default_keep_recent_messages")]
    pub keep_recent_tool_results: usize,
    #[serde(default)]
    pub clear_reasoning: bool,
    #[serde(default = "default_keep_recent_reasoning_turns")]
    pub keep_recent_reasoning_turns: usize,
}

impl Default for ConversationRule {
    fn default() -> Self {
        Self {
            keep_recent_messages: default_keep_recent_messages(),
            keep_recent_tool_results: default_keep_recent_messages(),
            clear_reasoning: false,
            keep_recent_reasoning_turns: default_keep_recent_reasoning_turns(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCompression {
    #[serde(default = "default_keep_recent_turns")]
    pub keep_recent_turns: usize,
    #[serde(default = "default_summary_layer_name")]
    pub summary_layer_name: String,
    #[serde(default = "default_summary_priority")]
    pub summary_priority: i32,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default = "default_include_existing_summary")]
    pub include_existing_summary: bool,
}

impl Default for ModelCompression {
    fn default() -> Self {
        Self {
            keep_recent_turns: default_keep_recent_turns(),
            summary_layer_name: default_summary_layer_name(),
            summary_priority: default_summary_priority(),
            instruction: None,
            include_existing_summary: default_include_existing_summary(),
        }
    }
}

pub struct GenerateSummaryModel<'a, M> {
    model: &'a M,
    request: GenerateRequest,
    system_prompt: String,
}

impl<'a, M> GenerateSummaryModel<'a, M> {
    pub fn new(model: &'a M, request: GenerateRequest) -> Self {
        Self {
            model,
            request,
            system_prompt: DEFAULT_SUMMARY_SYSTEM_PROMPT.to_string(),
        }
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = prompt.into();
        self
    }
}

#[async_trait]
impl<M> SummaryModel for GenerateSummaryModel<'_, M>
where
    M: GenerateCapability + Send + Sync,
{
    async fn summarize(&self, prompt: &str) -> Result<String, ModelError> {
        let response = self
            .model
            .generate(self.request.clone().with_messages(vec![
                Message::system(self.system_prompt.clone()),
                Message::user(prompt),
            ]))
            .await?;

        if response.message.role == MessageRole::Assistant {
            let text = response.message.text();
            if !text.trim().is_empty() {
                return Ok(text.into_owned());
            }
        }
        Err(ModelError::NoResponse)
    }
}

fn default_keep_recent_messages() -> usize {
    usize::MAX
}

fn default_keep_recent_reasoning_turns() -> usize {
    1
}

fn default_keep_recent_turns() -> usize {
    2
}

fn default_summary_layer_name() -> String {
    DEFAULT_SUMMARY_LAYER_NAME.to_string()
}

fn default_summary_priority() -> i32 {
    DEFAULT_SUMMARY_PRIORITY
}

fn default_include_existing_summary() -> bool {
    true
}
