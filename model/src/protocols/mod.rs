//! 协议层：wire 结构与统一多模态域类型之间的翻译。
//!
//! 一个协议适配器知道三件传输层不该知道的事：端点路径、请求/响应的 wire 形状、
//! 以及厂商私有字段怎么编码。它们统一实现 [`crate::capability::GenerateCapability`]，
//! 因此可以直接注册进 [`crate::router::ModelRouter`]。
//!
//! 这里**只放协议**。厂商的 `base_url` / 默认请求头 / 认证属于端点配置，在
//! [`crate::endpoints`]。协议的选择是 **(端点, 模态)** 的属性：
//!
//! | | 文本 / 视觉 / 音频 / 多模态对话 | 专用图片端点 |
//! |---|---|---|
//! | DeepSeek | [`openai`] | — |
//! | OpenRouter | [`openai`] | [`openrouter`] |

pub mod openai;
pub mod openrouter;

pub use openai::OpenAiCompatible;
pub use openrouter::OpenRouterImages;
