# alaya-agent-rs

> "What I cannot create, I do not understand."  
> —— Richard Feynman

一个用 Rust 编写的 LLM Agent 框架，从零实现了 Provider、Model Router、Tool、Context、Lifecycle Hook 等核心组件。

## 特性

- **OpenAI 兼容的 Provider 抽象** —— 内置 DeepSeek、OpenRouter 适配器，支持任何兼容 OpenAI API 的服务
- **流式 SSE 解析** —— 手写的 SSE 帧解析器，正确处理跨 chunk 边界的多字节字符和增量 tool call 合并
- **多模态路由** —— 按能力（Chat / Image / Audio）独立路由到不同 provider 和模型
- **分层上下文** —— System / Soul / User / Memory / Conversation / Custom 层，按优先级排序，支持合并与序列化
- **生命周期 Hook 系统** —— 7 个阶段（BeforeStep → BeforeCallModel → OnModelEvent → AfterCallModel → BeforeCallTools → AfterCallTools → AfterStep），可插拔扩展
- **工具调用** —— 并行执行、超时控制、JSON Schema 参数定义、自动注册
- **Actor 模型** —— 后台异步循环，支持暂停 / 恢复 / 取消 / 人工介入（ask_user）
- **上下文压缩** —— 规则压缩（drop / clear / trim / replace）和模型摘要压缩
- **完整指标追踪** —— 时间线、迭代次数、Token 用量、延迟、工具调用统计、错误记录
- **内置文件工具** —— file_list、file_search（regex grep）、file_read（按行范围）

## 架构

```
┌─────────────────────────────────────────────────────┐
│                    AgentActor                        │
│  ┌───────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Context   │  │ AgentState   │  │   Metrics    │  │
│  │ (分层上下文) │  │ (状态 + 指标) │  │ (运行统计)    │  │
│  └───────────┘  └──────────────┘  └──────────────┘  │
│         │                                           │
│    ┌────▼─────────────────────────────┐              │
│    │       StepLifeCycle (7 阶段)      │              │
│    │  BeforeStep → BeforeCallModel     │              │
│    │  → OnModelEvent → AfterCallModel  │              │
│    │  → BeforeCallTools → AfterCallTools│              │
│    │  → AfterStep                       │              │
│    └────┬──────────────┬───────────────┘              │
│         │              │                              │
│    ┌────▼────┐   ┌────▼─────┐                        │
│    │ ChatCap  │   │ToolExec  │                        │
│    │ (模型)   │   │ (工具)    │                        │
│    └────┬────┘   └──────────┘                        │
└─────────┼───────────────────────────────────────────┘
          │
    ┌─────▼─────┐
    │ModelRouter │  ──→  Chat / Image / Audio
    └─────┬─────┘
          │
    ┌─────▼──────────────────────────┐
    │         Provider               │
    │  ┌────────────┐ ┌───────────┐  │
    │  │ DeepSeek   │ │OpenRouter │  │
    │  │ (OpenAI    │ │(OpenAI    │  │
    │  │ Compatible)│ │Compatible)│  │
    │  └────────────┘ └───────────┘  │
    └────────────────────────────────┘
```

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
    context::{Context, Layer, LayerKind},
    providers::{Request, deepseek_provider_from_env},
    router::{ChatCapability, ModelCapability, ModelRouter},
    Message,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    // 1. 创建 Provider
    let provider = Arc::new(deepseek_provider_from_env()?);

    // 2. 配置模型路由
    let mut router = ModelRouter::new();
    router.add_model_provider("deepseek-chat", provider, &[ModelCapability::Chat]);

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
    let request = Request::new("deepseek-chat", messages);
    let response = router.chat(request).await?;

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
    providers::{Request, openrouter_provider_from_env},
    router::{ChatCapability, ModelCapability, ModelRouter},
    Message,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    let provider = Arc::new(openrouter_provider_from_env()?);
    let mut router = ModelRouter::new();
    router.add_model_provider("google/gemini-3-pro-preview", provider, &[ModelCapability::Chat]);

    let mut stream = router
        .chat_stream(
            Request::new(
                "google/gemini-3-pro-preview",
                vec![Message::user("从 1 数到 5")],
            )
            .with_stream(true),
        )
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

### Agent 循环 + 工具调用

```rust
use alaya_agent::{
    agent::{AgentActorBuilder, GenericToolExecutor, register_select_tools},
    context::{Context, Layer, LayerKind},
    providers::{Request, deepseek_provider_from_env},
    router::{ModelCapability, ModelRouter},
    select::SelectToolConfig,
};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenv::dotenv().ok();

    // Provider + Router
    let provider = Arc::new(deepseek_provider_from_env()?);
    let mut router = ModelRouter::new();
    router.add_model_provider("deepseek-chat", provider, &[ModelCapability::Chat]);

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
    let chat_request = Request::new("deepseek-chat", Vec::new()).with_stream(true);
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

## 模块说明

| 模块 | 说明 |
|------|------|
| `core` | 核心类型：`Message`、`Usage`、`MessageRole` |
| `providers` | LLM API Provider 抽象，`OpenAICompatibleProvider` 通用实现，DeepSeek / OpenRouter 适配器 |
| `router` | `ModelRouter` 按 Chat / Image / Audio 能力路由到不同模型和 Provider |
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
