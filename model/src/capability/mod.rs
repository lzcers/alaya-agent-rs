//! 能力层：面向调用方的统一多模态域接口。
//!
//! 上层（agent）只依赖这里的 [`GenerateCapability`] 与域类型，看不到任何
//! 协议结构体、端点路径或 HTTP 概念。实现此 trait 的是内部协议适配器
//! （[`crate::protocols`]）、端点组合体（[`crate::endpoints`]）与路由器
//! （[`crate::router::ModelRouter`]）。

mod error;
mod generate;
mod request;
mod response;

pub use error::ModelError;
pub use generate::{GenerateCapability, ModelProfile};
pub use request::{
    AudioOptions, GenerateConfig, GenerateRequest, ImageOptions, OutputSpec, ReasoningConfig,
    ReasoningEffort, TextFormat, ToolChoice,
};
pub use response::{
    FinishReason, GenerateResponse, GenerateStream, PartDelta, StreamAccumulator, StreamEvent,
    collect_stream,
};
