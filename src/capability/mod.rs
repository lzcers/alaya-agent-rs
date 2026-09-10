//! 能力层：面向调用方的域接口。
//!
//! 上层（agent）只依赖这里的 trait 与域类型，看不到任何 wire 结构体、
//! 端点路径或 HTTP 概念。实现这些 trait 的是协议适配器
//! （[`crate::protocols`]）与分发器（[`crate::router::ModelRouter`]）。

mod audio;
mod chat;
mod error;
mod image;

pub use audio::{GenAudioCapability, GenAudioRequest, GenAudioResponse};
pub use chat::{ChatCapability, ChatChunk, ChatRequest, Reasoning};
pub use error::{ModelError, format_model_error};
pub use image::{GenImgCapability, GenImgRequest, GenImgResponse};
