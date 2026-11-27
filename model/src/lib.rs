mod capability;
pub mod endpoints;
mod message;
mod protocols;
mod providers;
mod router;
mod transport;
mod usage;

pub use capability::{
    AudioOptions, FinishReason, GenerateCapability, GenerateConfig, GenerateRequest,
    GenerateResponse, GenerateStream, ImageOptions, ModelError, ModelProfile, OutputSpec,
    PartDelta, ReasoningConfig, ReasoningEffort, StreamAccumulator, StreamEvent, TextFormat,
    ToolChoice, collect_stream,
};
pub use message::{ContentPart, MediaSource, Message, MessageRole, Modality, ToolCall, ToolDef};
pub use router::{ModelRoute, ModelRouter};
pub use usage::Usage;

#[cfg(test)]
#[path = "../tests/unit/mod.rs"]
mod unit_tests;
