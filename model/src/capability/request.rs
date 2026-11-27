//! 生成请求与可复用配置域类型。
//!
//! - [`GenerateRequest`] 只承载单次调用载荷（`model` / `messages` / `tools`）与 [`GenerateConfig`]；
//! - [`GenerateConfig`] 承载可复用的生成参数与类型化输出规范 [`OutputSpec`]。

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{Message, Modality, ToolDef};

/// 推理强度档位。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// 厂商无关的推理配置。
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ReasoningConfig {
    /// 使用模型默认策略
    #[default]
    Default,
    /// 显式关闭推理
    Disabled,
    /// 显式开启推理
    Enabled,
    /// 按档位控制推理强度
    Effort { level: ReasoningEffort },
    /// 按 token 预算控制推理深度（Anthropic / Gemini）
    Budget { tokens: u32 },
}

/// 工具选择策略。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Tool {
        name: String,
    },
}

/// 文本输出格式。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextFormat {
    #[default]
    Plain,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(default)]
        strict: bool,
    },
}

/// 图片输出参数。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u32>,
}

impl ImageOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    pub fn with_count(mut self, count: u32) -> Self {
        self.count = Some(count);
        self
    }
}

/// 音频输出参数。
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
}

impl AudioOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }
}

/// 期望的输出模态及其专属参数（互斥且自洽，防止构造矛盾配置）。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "modality", rename_all = "snake_case")]
pub enum OutputSpec {
    Text(TextFormat),
    Image(ImageOptions),
    Audio(AudioOptions),
}

impl OutputSpec {
    pub fn text() -> Self {
        Self::Text(TextFormat::Plain)
    }

    pub fn json_object() -> Self {
        Self::Text(TextFormat::JsonObject)
    }

    pub fn json_schema(name: impl Into<String>, schema: Value, strict: bool) -> Self {
        Self::Text(TextFormat::JsonSchema {
            name: name.into(),
            schema,
            strict,
        })
    }

    pub fn image(options: ImageOptions) -> Self {
        Self::Image(options)
    }

    pub fn audio(options: AudioOptions) -> Self {
        Self::Audio(options)
    }

    pub fn modality(&self) -> Modality {
        match self {
            Self::Text(_) => Modality::Text,
            Self::Image(_) => Modality::Image,
            Self::Audio(_) => Modality::Audio,
        }
    }
}

/// 可复用的生成配置（输出规范、采样参数、推理策略与厂商扩展槽）。
///
/// 与单次调用载荷（`messages` / `tools`）分离，供 `AgentActor`、摘要压缩器及业务场景直接持有复用。
#[derive(Debug, Clone, Default)]
pub struct GenerateConfig {
    /// 期望的输出模态与专属参数；为空时默认纯文本（`[OutputSpec::text()]`）。
    pub outputs: Vec<OutputSpec>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<u32>,
    pub stop_sequences: Option<Vec<String>>,
    pub tool_choice: Option<ToolChoice>,
    pub reasoning: ReasoningConfig,
    /// 厂商私有参数逃生舱（协议层可选择性透传）。
    pub provider_options: Option<Value>,
}

impl GenerateConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_output(mut self, spec: OutputSpec) -> Self {
        self.outputs = vec![spec];
        self
    }

    pub fn with_outputs(mut self, specs: impl IntoIterator<Item = OutputSpec>) -> Self {
        self.outputs = specs.into_iter().collect();
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_stop_sequences(mut self, stops: Vec<String>) -> Self {
        self.stop_sequences = Some(stops);
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    pub fn with_reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = reasoning;
        self
    }

    pub fn with_provider_options(mut self, options: Value) -> Self {
        self.provider_options = Some(options);
        self
    }

    /// 实际生效的输出模态列表（`outputs` 为空时默认 `[Modality::Text]`）。
    pub fn output_modalities(&self) -> Vec<Modality> {
        if self.outputs.is_empty() {
            return vec![Modality::Text];
        }
        let mut out = Vec::with_capacity(self.outputs.len());
        for spec in &self.outputs {
            let modality = spec.modality();
            if !out.contains(&modality) {
                out.push(modality);
            }
        }
        out
    }

    pub fn wants_output(&self, modality: Modality) -> bool {
        if self.outputs.is_empty() {
            modality == Modality::Text
        } else {
            self.outputs.iter().any(|spec| spec.modality() == modality)
        }
    }

    /// 是否为纯图片生成请求（用于按端点路由到专用生图协议）。
    pub fn is_pure_image(&self) -> bool {
        matches!(self.outputs.as_slice(), [OutputSpec::Image(_)])
    }

    pub fn text_format(&self) -> Option<&TextFormat> {
        self.outputs.iter().find_map(|spec| match spec {
            OutputSpec::Text(format) => Some(format),
            _ => None,
        })
    }

    pub fn image_options(&self) -> Option<&ImageOptions> {
        self.outputs.iter().find_map(|spec| match spec {
            OutputSpec::Image(options) => Some(options),
            _ => None,
        })
    }

    pub fn audio_options(&self) -> Option<&AudioOptions> {
        self.outputs.iter().find_map(|spec| match spec {
            OutputSpec::Audio(options) => Some(options),
            _ => None,
        })
    }
}

/// 一次具体的多模态生成请求（调用载荷 + 生成配置）。
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    /// 逻辑模型名，同时作为分发层路由键。
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Option<Vec<ToolDef>>,
    pub config: GenerateConfig,
}

impl GenerateRequest {
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            tools: None,
            config: GenerateConfig::default(),
        }
    }

    /// 快捷构造：纯图片生成请求。
    pub fn image(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self::image_with(model, prompt, ImageOptions::default())
    }

    /// 快捷构造：带参数的纯图片生成请求。
    pub fn image_with(
        model: impl Into<String>,
        prompt: impl Into<String>,
        options: ImageOptions,
    ) -> Self {
        Self::new(model, vec![Message::user(prompt)]).with_output(OutputSpec::Image(options))
    }

    /// 快捷构造：音频生成请求。
    pub fn audio(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self::audio_with(model, prompt, AudioOptions::default())
    }

    /// 快捷构造：带参数的音频生成请求。
    pub fn audio_with(
        model: impl Into<String>,
        prompt: impl Into<String>,
        options: AudioOptions,
    ) -> Self {
        Self::new(model, vec![Message::user(prompt)]).with_output(OutputSpec::Audio(options))
    }

    pub fn with_messages(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
    }

    pub fn with_tools(mut self, tools: Option<Vec<ToolDef>>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_config(mut self, config: GenerateConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_output(mut self, spec: OutputSpec) -> Self {
        self.config = self.config.with_output(spec);
        self
    }

    pub fn with_outputs(mut self, specs: impl IntoIterator<Item = OutputSpec>) -> Self {
        self.config = self.config.with_outputs(specs);
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config = self.config.with_temperature(temperature);
        self
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.config = self.config.with_top_p(top_p);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.config = self.config.with_max_tokens(max_tokens);
        self
    }

    pub fn with_stop_sequences(mut self, stops: Vec<String>) -> Self {
        self.config = self.config.with_stop_sequences(stops);
        self
    }

    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.config = self.config.with_tool_choice(choice);
        self
    }

    pub fn with_reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.config = self.config.with_reasoning(reasoning);
        self
    }

    pub fn with_provider_options(mut self, options: Value) -> Self {
        self.config = self.config.with_provider_options(options);
        self
    }

    /// 汇总请求消息中实际出现的所有输入模态。
    pub fn input_modalities(&self) -> Vec<Modality> {
        let mut out = Vec::new();
        for message in &self.messages {
            for part in &message.content {
                part.collect_modalities(&mut out);
            }
        }
        out
    }

    /// 实际生效的输出模态列表。
    pub fn output_modalities(&self) -> Vec<Modality> {
        self.config.output_modalities()
    }
}
