# alaya-agent-rs

> "What I cannot create, I do not understand."  
> —— Richard Feynman

一个用 Rust 编写的 LLM Agent 框架，从零实现了传输层、厂商认证、协议层、统一多模态能力层、Model Router、Tool、Context、Lifecycle Hook 等核心组件。

## 特性

- **四层模型调用栈** —— 传输层（HTTP 字节流 + 完整 SSE 事件）/ 厂商层（地址、认证与签名钩子）/ 协议层（wire ↔ 域 IR 映射）/ 能力层（统一多模态 `GenerateCapability`）/ 分发层（能力画像校验 + 别名映射 + 主备切换）
- **统一多模态内容 IR** —— 所有角色（System / User / Assistant / Tool）共享统一的 `ContentPart` 与 `MediaSource`（Text / Image / Audio / Video / File / Reasoning+Signature / ToolCall / 多模态 ToolResult），支持任意模态输入与输出
- **事件化流式接口** —— `BoxStream<'static, Result<StreamEvent, ModelError>>` 显式传播流内错误，统一承载文本、推理、推理签名、音频流、图片与工具调用，并正确捕获尾部 `Usage`
- **多厂商接入友好** —— 内置 `Auth::Bearer` / `Header` / `QueryParam` / `Custom(RequestAuth)` 动态签名钩子；OpenAI 兼容厂商一行接入，自有协议厂商仅需实现 1 个 `GenerateCapability`
- **分层上下文** —— System / Soul / User / Memory / Conversation / Custom 层，按优先级排序，支持合并与序列化
- **生命周期 Hook 系统** —— 7 个阶段（BeforeStep → BeforeCallModel → OnModelEvent → AfterCallModel → BeforeCallTools → AfterCallTools → AfterStep），可插拔扩展
- **工具调用** —— 并行执行、超时控制、JSON Schema 参数定义、自动注册
- **Actor 模型** —— 后台异步循环，支持暂停 / 恢复 / 取消 / 人工介入（ask_user）
- **上下文压缩** —— 规则压缩（drop / clear / trim / replace）和模型摘要压缩
- **完整指标追踪** —— 时间线、迭代次数、Token 用量、延迟、工具调用统计、错误记录

## Workspace 结构

本项目是一个多 crate Cargo Workspace，底层模型调用栈与上层 Agent 编排完全解耦，可独立按需引入：

```text
├── model/   # alaya-model：transport / providers / protocols / capability / router / endpoints / message / usage
└── agent/   # alaya-agent（依赖 alaya-model）：AgentActor / Context / Hooks / Tools / Compress / Memory / Select
```

## 架构

```
┌──────────────────────────────────────────────────────┐
│                     AgentActor                        │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Context   │  │ AgentState   │  │   Metrics    │   │
│  │ (分层上下文) │  │ (状态 + 指标) │  │ (运行统计)    │   │
│  └───────────┘  └──────────────┘  └──────────────┘   │
│         │                                            │
│    ┌────▼─────────────────────────────┐              │
│    │       StepLifeCycle (7 阶段)      │              │
│    └────┬──────────────┬───────────────┘              │
│         │              │                              │
│    ┌────▼──────────┐ ┌─▼────────┐                     │
│    │GenerateCap    │ │ToolExec  │                     │
│    │(统一多模态域接口)│ │ (工具)    │                     │
│    └────┬──────────┘ └──────────┘                     │
└─────────┼────────────────────────────────────────────┘
          │
    ┌─────▼───────────────────────────┐
    │  ModelRouter                     │  分发层：逻辑名 → ModelRoute（上游名 + ModelProfile + 主备）
    └─────┬───────────────────────────┘
          │
    ┌─────▼───────────────────────────┐
    │  协议层（统一实现 GenerateCap）   │
    │  OpenAiCompatible  /chat/…       │  → 文本 / 视觉 / 音频 / 多模态对话
    │  OpenRouterImages  /images       │  → 专用图片端点
    └─────┬───────────────────────────┘
          │
    ┌─────▼───────────────────────────┐
    │  Provider（值）                   │  厂商身份：transport + base_url + headers + Auth
    └─────┬───────────────────────────┘
          │
    ┌─────▼───────────────────────────┐
    │  Transport（trait）               │  传输层：send(HttpRequest) -> HttpResponse
    │  HttpTransport + SSE(event/data) │  字节流收发，不认识厂商与业务
    └──────────────────────────────────┘
```

各层职责与依赖方向：

| 层 | 模块 | 形态 | 知道什么 | 不知道什么 |
|----|------|------|----------|------------|
| 分发层 | `router` | `ModelRouter` | 逻辑模型名 → 上游模型名 / `ModelProfile` / 适配器与主备 | 协议细节 |
| 能力层 | `capability` | **trait** | 统一域接口 `GenerateCapability`、`GenerateRequest`、`GenerateResponse`、`StreamEvent`、`ModelProfile`、`ModelError` | HTTP、JSON 形状 |
| 协议层 | `protocols` | 适配器 | 端点路径、wire 结构体、厂商字段与模态编码 | 模型路由、`base_url` |
| 厂商 | `providers` | **值** | `base_url`、默认请求头、`Auth`（Bearer / Header / Query / 动态签名） | 端点路径、wire 形状、路由 |
| 传输层 | `transport` | **trait** | 连接、超时、代理、完整 SSE 分帧（`event/data/id`）、原始字节流 | 厂商、密钥、业务操作 |
| 目录 | `endpoints` | 组合根 | 厂商地址、认证、把端点接到正确的协议上 | — |

依赖链：`router → capability ← protocols → providers → transport`。

## 快速开始

### 最简示例：文本对话（仅依赖 `alaya-model`）

```rust
use alaya_model::{
    Message,
    capability::{GenerateCapability, GenerateRequest, ModelProfile},
    endpoints::deepseek_from_env,
    router::ModelRouter,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(deepseek_from_env()?);
    let mut router = ModelRouter::new();
    router.register("deepseek-chat", ModelProfile::text(), model);

    let response = router
        .generate(GenerateRequest::new(
            "deepseek-chat",
            vec![Message::user("你好，介绍一下你自己。")],
        ))
        .await?;

    println!("{}", response.message.text());
    Ok(())
}
```

### 流式对话

```rust
use alaya_model::{
    Message,
    capability::{GenerateCapability, GenerateRequest, ModelProfile, PartDelta, StreamEvent},
    endpoints::openrouter_from_env,
    router::ModelRouter,
};
use futures::StreamExt;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(openrouter_from_env()?);
    let mut router = ModelRouter::new();
    router.register("google/gemini-2.5-flash", ModelProfile::omni(), model);

    let mut stream = router
        .generate_stream(GenerateRequest::new(
            "google/gemini-2.5-flash",
            vec![Message::user("从 1 数到 5")],
        ))
        .await?;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::Delta(PartDelta::Text { text }) => print!("{text}"),
            StreamEvent::Delta(PartDelta::Reasoning { text }) => eprint!("[推理] {text}"),
            _ => {}
        }
    }

    Ok(())
}
```

### 图片 / 音频生成（统一入口）

```rust
use alaya_model::{
    capability::{
        AudioOptions, GenerateCapability, GenerateRequest, ImageOptions, ModelProfile,
    },
    endpoints::openrouter_from_env,
    router::ModelRouter,
};
use std::sync::Arc;

let model = Arc::new(openrouter_from_env()?);
let mut router = ModelRouter::new();
router.register(
    "black-forest-labs/flux.2-klein-4b",
    ModelProfile::image_generation(),
    model.clone(),
);
router.register(
    "google/lyria-3-clip-preview",
    ModelProfile::audio_generation(),
    model,
);

// 生图：统一返回 ContentPart::Image(MediaSource)
let image_resp = router
    .generate(
        GenerateRequest::image("black-forest-labs/flux.2-klein-4b", "日落下的山峦")
            .with_image_options(ImageOptions::new().with_aspect_ratio("16:9").with_resolution("1K")),
    )
    .await?;

// 生音：统一返回 ContentPart::Audio(MediaSource + transcript)
let audio_resp = router
    .generate(
        GenerateRequest::audio("google/lyria-3-clip-preview", "一段轻快的钢琴循环")
            .with_audio_options(AudioOptions::new().with_format("wav")),
    )
    .await?;
```

### 接入新厂商

| 厂商情况 | 需要实现 | 代码 |
|----------|----------|------|
| **兼容 OpenAI** | **0 个 trait** | `endpoints::openai_compatible(name, key, base_url)` |
| **自有协议**（Anthropic / Gemini / Bedrock） | **1 个 `GenerateCapability` trait**；认证由 `Provider::with_header_auth` / `with_query_auth` / `with_auth(Auth::Custom(...))` 承担 | 见 `model/tests/custom_vendor.rs` |

## 模块说明

| Crate | 模块 | 说明 |
|-------|------|------|
| `alaya-model` | `message` / `usage` | 统一多模态域 IR：`Message`、`MessageRole`、`Modality`、`ContentPart`、`MediaSource`、`ToolDef`、`ToolCall`、`Usage` |
| `alaya-model` | `capability` | 统一能力层：`GenerateCapability`、`GenerateRequest`、`GenerateResponse`、`StreamEvent`、`StreamAccumulator`、`ModelProfile`、`ModelError` |
| `alaya-model` | `protocols` | 协议层：`OpenAiCompatible`、`OpenRouterImages` |
| `alaya-model` | `endpoints` | 端点预设：`openai_compatible`、`deepseek`、`openrouter` |
| `alaya-model` | `providers` | 厂商身份：`Provider`、`Auth`、`RequestAuth` |
| `alaya-model` | `transport` | 传输层：`Transport` trait、`HttpTransport`、`HttpRequest` / `HttpResponse`、`SseEvent`、`TransportError` |
| `alaya-model` | `router` | 分发层：`ModelRouter`、`ModelRoute`（能力画像校验、逻辑名/别名映射、主备切换） |
| `alaya-agent` | `agent_actor` / `context` / `hooks` / `tools` / `compress` / `memory` / `select` | Agent 编排层：`Context`、`AgentActor`、Lifecycle Hooks、工具执行、上下文压缩、记忆与文件选择 |

## 环境变量

| 变量 | 说明 | 必需 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 使用 DeepSeek 时 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址（默认 `https://api.deepseek.com`） | 否 |
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | 使用 OpenRouter 时 |
| `OPENROUTER_BASE_URL` | OpenRouter API 地址（默认 `https://openrouter.ai/api/v1`） | 否 |
| `OPENROUTER_HTTP_REFERER` | OpenRouter HTTP-Referer 头 | 否 |
| `OPENROUTER_X_TITLE` | OpenRouter X-Title 头 | 否 |

## License

MIT
