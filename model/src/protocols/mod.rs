//! 协议层：目标协议的数据结构定义与 [`crate::capability::GenerateCapability`] 适配。
//!
//! 每个子模块分两层：
//! 1. `types`：按目标协议规范定义请求/响应数据结构（不依赖域模型）；
//! 2. 适配器：负责端点路径、域 ↔ 协议双向映射，并实现 [`crate::capability::GenerateCapability`]，
//!    因此可以直接注册进 [`crate::router::ModelRouter`]。
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
