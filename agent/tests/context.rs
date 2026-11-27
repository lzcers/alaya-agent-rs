use alaya_agent::{Context, Layer, LayerKind};
use alaya_model::{Message, MessageRole};
use serde_json::Value;

#[test]
fn test_context_layer() {
    let ctx = Context::new()
        .layer(
            Layer::new(
                "system",
                LayerKind::System,
                Value::String("You are helpful.".into()),
            )
            .with_priority(100),
        )
        .layer(
            Layer::new(
                "soul",
                LayerKind::Soul,
                serde_json::json!({
                    "name": "Kśana",
                    "role": "AI Assistant",
                    "guidelines": ["Be helpful", "Be concise"]
                }),
            )
            .with_priority(50),
        );

    assert_eq!(ctx.layers.len(), 2);

    let system = ctx.get("system").unwrap();
    assert_eq!(system.meta.priority, 100);
}

#[test]
fn test_context_to_messages() {
    let ctx = Context::new()
        .layer(Layer::new(
            "system",
            LayerKind::System,
            Value::String("Be helpful.".into()),
        ))
        .layer(Layer::new(
            "conversation",
            LayerKind::Conversation,
            serde_json::to_value(vec![Message::user("Hello"), Message::assistant("Hi!")]).unwrap(),
        ));

    let messages = ctx.to_messages();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, MessageRole::System);
}

#[test]
fn test_memory_layers_become_user_messages() {
    let ctx = Context::new()
        .layer(
            Layer::new(
                "system",
                LayerKind::System,
                Value::String("Be helpful.".into()),
            )
            .with_priority(100),
        )
        .layer(
            Layer::new(
                "history",
                LayerKind::Memory,
                serde_json::json!([{ "content": "世界历史摘要\n王国陷入内战。" }]),
            )
            .with_priority(60),
        )
        .layer(
            Layer::new(
                "state",
                LayerKind::Memory,
                serde_json::json!([{ "content": "当前状态\n位置：王都广场" }]),
            )
            .with_priority(40),
        );

    let messages = ctx.to_messages();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].role, MessageRole::System);
    assert_eq!(messages[1].role, MessageRole::User);
    assert_eq!(messages[2].role, MessageRole::User);
    assert_eq!(messages[1].text(), "世界历史摘要\n王国陷入内战。");
    assert_eq!(messages[2].text(), "当前状态\n位置：王都广场");
}

#[test]
fn test_context_add_message() {
    let mut ctx = Context::new();
    ctx.add_message(Message::user("Hello"));
    ctx.add_message(Message::assistant("Hi!"));

    let conv = ctx.conversation();
    assert_eq!(conv.len(), 2);
}

#[test]
fn test_context_rollback_latest_input_removes_last_turn() {
    let mut ctx = Context::new().layer(Layer::new(
        "conversation",
        LayerKind::Conversation,
        serde_json::to_value(vec![
            Message::user("turn 1"),
            Message::assistant("answer 1"),
            Message::user("turn 2"),
            Message::assistant(""),
            Message::tool_result("call_1", "tool result"),
            Message::assistant("answer 2"),
        ])
        .unwrap(),
    ));

    assert!(ctx.rollback_latest_input());

    let conv = ctx.conversation();
    assert_eq!(conv.len(), 2);
    assert_eq!(conv[0].text(), "turn 1");
    assert_eq!(conv[1].text(), "answer 1");
}

#[test]
fn test_context_rollback_latest_input_removes_pending_user_message() {
    let mut ctx = Context::new().layer(Layer::new(
        "conversation",
        LayerKind::Conversation,
        serde_json::to_value(vec![
            Message::user("turn 1"),
            Message::assistant("answer 1"),
            Message::user("pending input"),
        ])
        .unwrap(),
    ));

    assert!(ctx.rollback_latest_input());

    let conv = ctx.conversation();
    assert_eq!(conv.len(), 2);
    assert_eq!(conv[0].text(), "turn 1");
    assert_eq!(conv[1].text(), "answer 1");
}

#[test]
fn test_context_rollback_latest_input_returns_false_without_user_message() {
    let mut ctx = Context::new().layer(Layer::new(
        "conversation",
        LayerKind::Conversation,
        serde_json::to_value(vec![Message::assistant("answer only")]).unwrap(),
    ));

    assert!(!ctx.rollback_latest_input());
    assert_eq!(ctx.conversation().len(), 1);
}

#[test]
fn test_context_print() {
    let ctx = Context::new()
        .layer(Layer::new(
            "system",
            LayerKind::System,
            Value::String("Be helpful.".into()),
        ))
        .layer(Layer::new(
            "conversation",
            LayerKind::Conversation,
            serde_json::to_value(vec![Message::user("Hello"), Message::assistant("Hi!")]).unwrap(),
        ));

    assert_eq!(
        ctx.print(),
        "system: \nBe helpful.\nuser: \nHello\nassistant: \nHi!"
    );
}

#[test]
fn test_context_print_latest_msg() {
    let ctx = Context::new().layer(Layer::new(
        "conversation",
        LayerKind::Conversation,
        serde_json::to_value(vec![Message::user("Hello"), Message::assistant("Hi!")]).unwrap(),
    ));

    assert_eq!(ctx.print_latest_msg().as_deref(), Some("assistant: \nHi!"));
}
