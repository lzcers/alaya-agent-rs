//! 协议层：wire 结构与域类型之间的翻译。
//!
//! 一个协议适配器知道三件传输层不该知道的事：端点路径、请求/响应的 wire 形状、
//! 以及厂商私有字段怎么编码。它实现能力层 trait，因此可以被直接注册进
//! [`crate::router::ModelRouter`]。
//!
//! 这里**只放协议**。厂商的 `base_url` / 默认请求头属于端点配置，在
//! [`crate::endpoints`]。协议的选择是 **(端点, 能力)** 的属性：
//!
//! | | chat | image | audio |
//! |---|---|---|---|
//! | DeepSeek | [`openai`] | — | — |
//! | OpenRouter | [`openai`] | [`openrouter`] | [`openai`] |

pub mod openai;
pub mod openrouter;

pub use openai::OpenAiCompatible;
pub use openrouter::OpenRouterImages;
