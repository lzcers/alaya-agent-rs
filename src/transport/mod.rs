//! 传输层：把字节发出去，把字节拿回来。
//!
//! 这一层只有一个抽象 [`Transport`]，它不知道厂商、密钥和业务语义：
//! URL、请求头、请求体都是调用方给的参数，响应体永远是字节流。
//!
//! 厂商身份在 [`crate::providers`]，wire 形状在 [`crate::protocols`]，
//! 域接口在 [`crate::capability`]。
//!
//! ```text
//! router → capability ← protocols → providers → transport
//! ```

mod error;
mod http;
mod request;
pub mod sse;

pub use error::TransportError;
pub use http::{HttpTransport, Transport};
pub use request::{BodyStream, HttpRequest, HttpResponse};
