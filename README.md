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
    │  Provider (post_json / post_sse) │  传输层：URL / 认证 / 超时 / 代理 / SSE 分帧
    │      → HttpProvider (reqwest)    │
    └──────────────────────────────────┘
```

各层职责与依赖方向：

| 层 | 模块 | 知道什么 | 不知道什么 |
|----|------|----------|------------|
| 传输层 | `providers` | URL、认证、超时、代理、SSE 分帧、HTTP 错误 | 任何业务操作（没有 `chat` / `generate_image` 方法） |
| 协议层 | `protocols` | 端点路径、wire 结构体、厂商字段编码 | 模型名怎么路由、`base_url` |
| 端点 | `endpoints` | `base_url`、默认请求头、厂商命名 | 协议形状、模型路由 |
| 能力层 | `capability` | 域接口与域类型（`ChatRequest`、`ChatChunk`、`GenImgResponse`……） | HTTP、JSON 形状 |
| 分发层 | `router` | 模型名 → 能力实现的映射 | 协议细节 |

`protocols` 里只有协议：DeepSeek 与 OpenRouter 的 chat/audio 都走 OpenAI 兼容协议，只有 OpenRouter 的图片端点走它自有的统一 schema。厂商的 `base_url` 与默认请求头属于端点配置，放在 `endpoints`。

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

### 自定义传输

`providers::Provider` 只有两个协议动词，实现它就能把模型调用接到任意 HTTP 栈上
（测试替身、录制回放、自定义中间件）：

```rust
use alaya_agent::providers::{Provider, ProviderError};
use futures::stream::BoxStream;
use serde_json::Value;

struct MyTransport;

#[async_trait::async_trait]
impl Provider for MyTransport {
    async fn post_json(&self, path: &str, body: Value) -> Result<Value, ProviderError> {
        // POST base_url + path
        todo!()
    }

    async fn post_sse(
        &self,
        path: &str,
        body: Value,
    ) -> Result<BoxStream<'static, Result<String, ProviderError>>, ProviderError> {
        // POST 后把响应体切成 SSE 事件的 data 载荷
        todo!()
    }

    fn name(&self) -> &str {
        "my-transport"
    }
}
```

## 模块说明

| 模块 | 说明 |
|------|------|
| `message` / `usage` | 核心域类型：`Message`、`MessageRole`、`Usage` |
| `capability` | 能力层：`ChatCapability` / `GenImgCapability` / `GenAudioCapability` 及域请求/响应类型、`ModelError` |
| `protocols` | 协议层：`OpenAiCompatible`（OpenAI 兼容 chat/audio）、`OpenRouterImages`（OpenRouter 图片 API） |
| `endpoints` | 端点预设：`deepseek` / `openrouter` 构造函数（`base_url` + 默认请求头），把能力接到正确的协议上 |
| `providers` | 传输层：`Provider` trait（`post_json` / `post_sse` / `name`）与 `HttpProvider` 实现、`ProviderError` |
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
