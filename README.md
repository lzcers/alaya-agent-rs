# alaya-agent-rs

> "What I cannot create, I do not understand."  
> —— Richard Feynman

一个用 Rust 编写的 LLM Agent 框架，从零实现了传输层、协议层、能力层、Model Router、Tool、Context、Lifecycle Hook 等核心组件。

## 特性

- **四层模型调用栈** —— 传输层（HTTP 协议动词）/ 协议层（wire ↔ 域映射）/ 能力层（域接口）/ 分发层（模型名路由）各司其职
- **OpenAI 兼容协议适配器** —— 内置 DeepSeek、OpenRouter 构造函数，支持任何兼容 OpenAI API 的服务
- **流式 SSE 解析** —— 手写的 SSE 帧解析器，正确处理跨 chunk 边界的多字节字符和增量 tool call 合并
- **多模态路由** —— 按能力（Chat / Image / Audio）独立路由到不同实现和模型
- **分层上下文** —— System / Soul / User / Memory / Conversation / Custom 层，按优先级排序，支持合并与序列化
- **生命周期 Hook 系统** —— 7 个阶段（BeforeStep → BeforeCallModel → OnModelEvent → AfterCallModel → BeforeCallTools → AfterCallTools → AfterStep），可插拔扩展
- **工具调用** —— 并行执行、超时控制、JSON Schema 参数定义、自动注册
- **Actor 模型** —— 后台异步循环，支持暂停 / 恢复 / 取消 / 人工介入（ask_user）
- **上下文压缩** —— 规则压缩（drop / clear / trim / replace）和模型摘要压缩
- **完整指标追踪** —— 时间线、迭代次数、Token 用量、延迟、工具调用统计、错误记录
- **内置文件工具** —— file_list、file_search（regex grep）、file_read（按行范围）

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
│    │  BeforeStep → BeforeCallModel     │              │
│    │  → OnModelEvent → AfterCallModel  │              │
│    │  → BeforeCallTools → AfterCallTools│              │
│    │  → AfterStep                       │              │
│    └────┬──────────────┬───────────────┘              │
│         │              │                              │
│    ┌────▼────┐   ┌────▼─────┐                         │
│    │ChatCap  │   │ToolExec  │                         │
│    │(域接口) │   │ (工具)    │                         │
│    └────┬────┘   └──────────┘                         │
└─────────┼────────────────────────────────────────────┘
          │
    ┌─────▼──────┐
    │ModelRouter │  分发层：按 (能力, 模型名) 查表转发
    └─────┬──────┘
          │
    ┌─────▼───────────────────────────┐
    │  协议层（协议是 (端点,能力) 的属性）│
    │  OpenAiCompatible  /chat/…       │  → Chat / Audio
    │  OpenRouterImages  /images       │  → Image
    └─────┬───────────────────────────┘
          │
    ┌─────▼───────────────────────────┐
    │  Provider                        │  厂商身份：transport + base_url + headers
    │   （值，不是 trait）              │  认证方案在这里，不在传输层
    └─────┬───────────────────────────┘
          │
    ┌─────▼───────────────────────────┐
    │  Transport                       │  传输层：send(HttpRequest) -> HttpResponse
    │  HttpTransport (reqwest)         │  连接 / 超时 / 代理 / SSE 分帧，不认识厂商
    └──────────────────────────────────┘
```

各层职责与依赖方向：

| 层 | 模块 | 形态 | 知道什么 | 不知道什么 |
|----|------|------|----------|------------|
| 分发层 | `router` | `ModelRouter` | 模型名 → 能力实现的映射 | 协议细节 |
| 能力层 | `capability` | **trait** | 域接口与域类型（`ChatRequest`、`ChatChunk`、`GenImgResponse`……） | HTTP、JSON 形状 |
| 协议层 | `protocols` | 适配器 | 端点路径、wire 结构体、厂商字段编码 | 模型名怎么路由、`base_url` |
| 厂商 | `providers` | **值** | `base_url`、请求头、认证方案 | 端点路径、wire 形状、路由 |
| 传输层 | `transport` | **trait** | 连接、超时、代理、SSE 分帧、HTTP 状态 | 厂商、密钥、业务操作 |
| 目录 | `endpoints` | 组合根 | 厂商地址、命名、把能力接到正确的协议上 | — |

`protocols` 里只有协议：DeepSeek 与 OpenRouter 的 chat/audio 都走 OpenAI 兼容协议，只有 OpenRouter 的图片端点走它自有的统一 schema。厂商的 `base_url` 与请求头属于 `Provider`（一个值，不是 trait——厂商之间变化的是数据，不是行为）。

依赖链：`router → capability ← protocols → providers → transport`。只有 `Transport` 是 trait（可以换实现：reqwest / 测试替身 / 其他栈）；越往上越具体。

新增一种能力（embeddings、rerank、TTS……）只需在 `capability` 加一个 trait、在 `protocols` 加一个实现，传输层与分发层不用改。

## 快速开始

### 添加依赖

```toml
[dependencies]
alaya-agent = "0.1.0"
tokio = { version = "1", features = ["full"] }
```

### 最简示例：一次对话

```rust
use alaya_agent::{
    agent::{Context, Layer, LayerKind},
    capability::{ChatCapability, ChatRequest},
    endpoints::deepseek_from_env,
    router::ModelRouter,
    Message,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    // 1. 创建端点（内部组装传输 + 协议）
    let model = Arc::new(deepseek_from_env()?);

    // 2. 注册 chat 能力
    let mut router = ModelRouter::new();
    router.add_chat_model("deepseek-chat", model);

    // 3. 构建上下文
    let ctx = Context::new().layer(Layer::new(
        "system",
        LayerKind::System,
        serde_json::Value::String("你是一个简洁的助手。".into()),
    ));

    let messages = {
        let mut ctx = ctx;
        ctx.add_message(Message::user("你好，介绍一下你自己。"));
        ctx.to_messages()
    };

    // 4. 调用模型
    let response = router
        .chat(ChatRequest::new("deepseek-chat", messages))
        .await?;

    if let Message::Assistant { content, .. } = response {
        println!("{}", content);
    }

    Ok(())
}
```

### 流式对话

```rust
use futures::StreamExt;
use alaya_agent::{
    capability::{ChatCapability, ChatRequest},
    endpoints::openrouter_from_env,
    router::ModelRouter,
    Message,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    let model = Arc::new(openrouter_from_env()?);
    let mut router = ModelRouter::new();
    router.add_chat_model("google/gemini-3-pro-preview", model);

    let mut stream = router
        .chat_stream(ChatRequest::new(
            "google/gemini-3-pro-preview",
            vec![Message::user("从 1 数到 5")],
        ))
        .await?;

    while let Some(chunk) = stream.next().await {
        print!("{}", chunk.content);
        if !chunk.reasoning_content.is_empty() {
            eprint!("[推理] {}", chunk.reasoning_content);
        }
    }

    Ok(())
}
```

### 图片 / 音频生成

```rust
use alaya_agent::{
    capability::{GenAudioCapability, GenAudioRequest, GenImgCapability, GenImgRequest},
    endpoints::openrouter_from_env,
    router::ModelRouter,
};
use std::sync::Arc;

let model = Arc::new(openrouter_from_env()?);
let mut router = ModelRouter::new();
router.add_image_model("black-forest-labs/flux.2-klein-4b", model.clone());
router.add_audio_model("google/lyria-3-clip-preview", model);

// 图片：返回值已统一成 URL（远程地址或 data: 内联数据）
let image = router
    .gen_img(
        GenImgRequest::new("black-forest-labs/flux.2-klein-4b", "日落下的山峦")
            .with_aspect_ratio("16:9")
            .with_resolution("1K"),
    )
    .await?;

// 音频：域请求只表达意图，modalities / audio 字段由协议层负责编码
let audio = router
    .gen_audio(GenAudioRequest::new("google/lyria-3-clip-preview", "一段轻快的钢琴循环").with_format("wav"))
    .await?;
```

### Agent 循环 + 工具调用

```rust
use alaya_agent::{
    agent::{AgentActorBuilder, GenericToolExecutor, register_select_tools},
    agent::{Context, Layer, LayerKind},
    capability::ChatRequest,
    endpoints::deepseek_from_env,
    router::ModelRouter,
    select::SelectToolConfig,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    // 协议适配器 + Router
    let model = Arc::new(deepseek_from_env()?);
    let mut router = ModelRouter::new();
    router.add_chat_model("deepseek-chat", model);

    // 工具执行器（注册文件操作工具）
    let mut executor = GenericToolExecutor::new();
    register_select_tools(&mut executor, SelectToolConfig::new("."));

    // 上下文
    let ctx = Context::new().layer(Layer::new(
        "system",
        LayerKind::System,
        serde_json::Value::String("你是一个代码助手，可以读写文件。".into()),
    ));

    // 构建 Agent
    let chat_request = ChatRequest::new("deepseek-chat", Vec::new());
    let agent = AgentActorBuilder::new(router, chat_request, executor)
        .context(ctx)
        .max_iterations(20)
        .build();

    // 启动后台循环，获取控制句柄
    let handle = agent.run_loop();

    // 等待完成，收集所有事件
    let events = handle.wait().await;

    for event in &events {
        println!("{:?}", event);
    }

    Ok(())
}
```

### 人工介入（ask_user）

```rust
use alaya_agent::{agent::AgentActorCommand};

// Agent 在执行中通过 ask_user 工具发起提问
// 外部通过 handle 响应：
handle.provide_input("用户的回答".into(), "input_id".into()).await;

// 控制操作
handle.pause().await;
handle.resume().await;
handle.cancel().await;
```

### 接入新厂商

| 厂商情况 | 需要实现 | 代码 |
|----------|----------|------|
| **兼容 OpenAI**（chat / audio） | **0 个 trait** | `endpoints::openai_compatible(name, key, base_url)` |
| **兼容 OpenAI 且提供生图** | 目前需自己实现 `GenImgCapability`（内置只覆盖 OpenRouter 的图片形状） | 见 `tests/custom_vendor.rs` |
| **自有协议**（Anthropic / Gemini） | 1 个 capability trait；认证非 Bearer 时再加 `Provider` | 见 `tests/custom_vendor.rs` |

```rust
use alaya_agent::{endpoints::openai_compatible, router::ModelRouter};
use std::sync::Arc;

// 定义 base_url 与 key —— 这就是全部接入代码
let model = Arc::new(openai_compatible(
    "groq",
    "your-api-key",
    "https://api.groq.com/openai/v1",
));

let mut router = ModelRouter::new();
router.add_chat_model("llama-3.3-70b-versatile", model.clone());
router.add_audio_model("some-audio-model", model);
```

「声明它兼容 OpenAI」体现在两处类型化的选择上，而不是一个字符串：

1. 用 `OpenAiCompatible` 作为适配器 → 声明走 OpenAI 协议；
2. 注册到 `add_chat_model` / `add_audio_model` / `add_image_model` → 声明这个端点提供哪些能力。

`tests/custom_vendor.rs` 是一份可执行的接入模板，两条路径都有完整示例。

### 自定义传输

`transport::Transport` 只有一个方法：**发一次请求**。响应体永远是字节流，怎么解释由调用方决定——所以二进制接口（TTS 的裸音频、文件上传）不需要给它加新方法。

只在换掉 HTTP 栈时才需要实现它（测试替身、录制回放、gRPC/WebSocket）。**认证方案不需要自定义传输**——那是 `Provider::with_header` 的事：

```rust
use alaya_agent::transport::{HttpRequest, HttpResponse, Transport, TransportError};

struct MyTransport;

#[async_trait::async_trait]
impl Transport for MyTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, TransportError> {
        // request.url / request.headers / request.body 都已由 Provider 备好
        todo!()
    }
}
```

```rust
use alaya_agent::providers::Provider;
use alaya_agent::transport::HttpTransport;
use std::sync::Arc;

// 非 Bearer 认证：换个头就行，不需要写传输层
let provider = Provider::new(Arc::new(HttpTransport::new()), "anthropic", "https://api.anthropic.com/v1")
    .with_header("x-api-key", api_key)
    .with_header("anthropic-version", "2023-06-01");
```

## 模块说明

| 模块 | 说明 |
|------|------|
| `message` / `usage` | 核心域类型：`Message`、`MessageRole`、`Usage` |
| `capability` | 能力层：`ChatCapability` / `GenImgCapability` / `GenAudioCapability` 及域请求/响应类型、`ModelError` |
| `protocols` | 协议层：`OpenAiCompatible`（OpenAI 兼容 chat/audio）、`OpenRouterImages`（OpenRouter 图片 API） |
| `endpoints` | 端点预设：`deepseek` / `openrouter` 构造函数（`base_url` + 默认请求头），把能力接到正确的协议上 |
| `transport` | 传输层：`Transport` trait（`send`）、`HttpTransport`、`HttpRequest` / `HttpResponse`、SSE 分帧、`TransportError` |
| `providers` | 厂商身份：`Provider`（transport + `base_url` + 请求头），一个值而非 trait |
| `router` | 分发层：`ModelRouter` 按 (能力, 模型名) 查表转发 |
| `agent::context` | 分层上下文容器 `Context`，支持 System / Soul / User / Memory / Conversation / Custom 层 |
| `agent::agent_actor` | `AgentActor` 组合模型 + 工具，执行单步或后台循环；`AgentActorBuilder` 构建器 |
| `agent::hooks` | 生命周期 Hook trait，内置 `ExecutionPolicyHook`、`MetricsHook`、`AskUserHook` 等 |
| `agent::tools` | `Tool` trait、`ToolRegistry`、`GenericToolExecutor`；OpenAI 兼容的工具定义和调用 |
| `agent::compress` | 上下文压缩：规则压缩（drop/clear/trim/replace）和模型摘要压缩 |
| `agent::memory` | 持久化记忆存储 `MemoryStore`，支持文件读写、目录列举、路径遍历防护 |
| `agent::select` | 文件选择工具集：`file_list`、`file_search`、`file_read` |
| `agent::filesystem` | `FsMemoryStore` 文件系统记忆后端、`FsSelector` 文件选择器 |

## 环境变量

| 变量 | 说明 | 必需 |
|------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 使用 DeepSeek 时 |
| `DEEPSEEK_BASE_URL` | DeepSeek API 地址（默认 `https://api.deepseek.com`） | 否 |
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | 使用 OpenRouter 时 |
| `OPENROUTER_BASE_URL` | OpenRouter API 地址（默认 `https://openrouter.ai/api/v1`） | 否 |
| `OPENROUTER_HTTP_REFERER` | OpenRouter HTTP-Referer 头 | 否 |
| `OPENROUTER_X_TITLE` | OpenRouter X-Title 头 | 否 |

## 运行测试

```bash
# 单元测试（不需要 API 密钥）
cargo test

# 集成测试（需要设置对应的环境变量）
DEEPSEEK_API_KEY=xxx cargo test -- --ignored
```

## License

MIT
