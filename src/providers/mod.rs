//! 传输层。
//!
//! 只做一件事：把 JSON 请求发出去，把 JSON 响应或 SSE 事件流拿回来。
//! 这里没有 `chat`、`generate_image` 之类的业务方法——那些属于能力层与协议层。
//!
//! 分层约定：
//! - 传输层（本模块）：URL、认证、超时、代理、SSE 分帧、HTTP 错误映射。
//! - 协议层（[`crate::protocols`]）：端点路径、wire 结构体、wire ↔ 域类型映射。
//! - 能力层（[`crate::capability`]）：面向调用方的域接口与域类型。
//! - 分发层（[`crate::router`]）：模型名 → 能力实现的映射。

mod error;
mod http;
mod sse;

pub use error::{ProviderError, parse_api_error};
pub use http::{HttpProvider, Provider};
