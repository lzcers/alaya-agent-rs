use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};

use crate::{
    Message, MessageRole, Usage,
    agent::ToolCall,
    providers::{Request, Response},
    router::{ModelCapability, ModelRouter, RouterError},
};

/// A streamed chat response chunk.
#[derive(Debug, Clone)]
pub struct ChatChunk {
    pub content: String,
    pub reasoning_content: String,
    pub is_finished: bool,
    pub finish_reason: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub usage: Option<Usage>,
}

#[async_trait]
pub trait ChatCapability {
    async fn chat(&self, request: Request) -> Result<Message, RouterError>;

    async fn chat_stream(
        &self,
        request: Request,
    ) -> Result<BoxStream<'static, ChatChunk>, RouterError>;
}

#[async_trait]
impl ChatCapability for ModelRouter {
    async fn chat(&self, request: Request) -> Result<Message, RouterError> {
        let provider = self.route(&request.model, ModelCapability::Chat)?;
        let response: Response = provider.chat(request).await?;

        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or(RouterError::NoResponse)?;

        match choice.message.role {
            MessageRole::Assistant => Ok(Message::Assistant {
                content: choice.message.content.unwrap_or_default(),
                reasoning_content: choice.message.reasoning_content,
                tool_calls: choice.message.tool_calls,
            }),
            MessageRole::User => Ok(Message::User {
                content: choice.message.content.unwrap_or_default(),
            }),
            MessageRole::System => Ok(Message::System {
                content: choice.message.content.unwrap_or_default(),
            }),
            MessageRole::Tool => Ok(Message::Tool {
                tool_call_id: choice.message.tool_call_id.unwrap_or_default(),
                content: choice.message.content.unwrap_or_default(),
            }),
        }
    }

    async fn chat_stream(
        &self,
        request: Request,
    ) -> Result<BoxStream<'static, ChatChunk>, RouterError> {
        let provider = self.route(&request.model, ModelCapability::Chat)?;
        let stream = provider.chat_stream(request).await?;

        Ok(stream
            .map(|response| {
                if let Some(choice) = response.choices.first() {
                    let content = choice.delta.content.clone().unwrap_or_default();
                    let reasoning_content =
                        choice.delta.reasoning_content.clone().unwrap_or_default();
                    let is_finished = choice.finish_reason.is_some();
                    ChatChunk {
                        content,
                        reasoning_content,
                        is_finished,
                        finish_reason: choice.finish_reason.clone(),
                        tool_calls: choice.delta.tool_calls.clone(),
                        usage: response.usage.map(Into::into),
                    }
                } else {
                    ChatChunk {
                        content: String::new(),
                        reasoning_content: String::new(),
                        is_finished: true,
                        finish_reason: Some("no_choices".to_string()),
                        tool_calls: None,
                        usage: response.usage.map(Into::into),
                    }
                }
            })
            .boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;
    use crate::providers::{
        deepseek_provider, deepseek_provider_from_env, openrouter_provider,
        openrouter_provider_from_env,
    };
    use serde_json::json;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_chat_with_deepseek_chat() {
        dotenv::dotenv().ok();

        let provider = match deepseek_provider_from_env() {
            Ok(p) => Arc::new(p),
            Err(_) => {
                eprintln!("DEEPSEEK_API_KEY not set, skipping test");
                return;
            }
        };

        let mut model = ModelRouter::new();
        model.add_models_for_provider(
            &["deepseek-chat", "deepseek-reasoner"],
            provider,
            &[ModelCapability::Chat],
        );

        let msg = Message::user("Say 'Hello, world!' in one sentence.");

        let result = model.chat(Request::new("deepseek-chat", vec![msg])).await;
        assert!(result.is_ok());

        let message = result.unwrap();
        if let Message::Assistant { content, .. } = message {
            println!("Response: {:?}", content);
            assert!(!content.is_empty());
        } else {
            panic!("Expected Assistant message");
        }
    }

    #[tokio::test]
    async fn test_chat_stream_with_deepseek_chat() {
        dotenv::dotenv().ok();

        let provider = match deepseek_provider_from_env() {
            Ok(p) => Arc::new(p),
            Err(_) => {
                eprintln!("DEEPSEEK_API_KEY not set, skipping test");
                return;
            }
        };

        let mut model = ModelRouter::new();
        model.add_models_for_provider(
            &["deepseek-chat", "deepseek-reasoner"],
            provider,
            &[ModelCapability::Chat],
        );

        let msg = Message::user("Count from 1 to 3, each number on a new line.");

        let result = model
            .chat_stream(Request::new("deepseek-chat", vec![msg]).with_stream(true))
            .await;
        assert!(result.is_ok());

        let mut stream = result.unwrap();
        let mut full_content = String::new();

        while let Some(chunk) = stream.next().await {
            print!("{}", chunk.content);
            full_content.push_str(&chunk.content);
            if chunk.is_finished {
                println!("\nFinish reason: {:?}", chunk.finish_reason);
            }
        }

        assert!(!full_content.is_empty());
    }

    #[tokio::test]
    async fn test_chat_with_deepseek_reasoner() {
        dotenv::dotenv().ok();

        let provider = match deepseek_provider_from_env() {
            Ok(p) => Arc::new(p),
            Err(_) => {
                eprintln!("DEEPSEEK_API_KEY not set, skipping test");
                return;
            }
        };

        let mut model = ModelRouter::new();
        model.add_models_for_provider(
            &["deepseek-chat", "deepseek-reasoner"],
            provider,
            &[ModelCapability::Chat],
        );

        let msg = Message::user("What is 15 + 27? Please think step by step.");

        let result = model
            .chat(Request::new("deepseek-reasoner", vec![msg]))
            .await;
        assert!(result.is_ok());

        let message = result.unwrap();
        if let Message::Assistant {
            content,
            reasoning_content,
            ..
        } = message
        {
            println!("Response: {:?}", content);
            assert!(!content.is_empty());
            // 推理模型应该返回推理内容
            if let Some(rc) = reasoning_content {
                println!("Reasoning content length: {}", rc.len());
                assert!(!rc.is_empty(), "Reasoning content should not be empty");
            }
        } else {
            panic!("Expected Assistant message");
        }
    }

    #[tokio::test]
    async fn test_chat_stream_with_deepseek_reasoner() {
        dotenv::dotenv().ok();

        let provider = match deepseek_provider_from_env() {
            Ok(p) => Arc::new(p),
            Err(_) => {
                eprintln!("DEEPSEEK_API_KEY not set, skipping test");
                return;
            }
        };

        let mut model = ModelRouter::new();
        model.add_models_for_provider(
            &["deepseek-chat", "deepseek-reasoner"],
            provider,
            &[ModelCapability::Chat],
        );

        let msg = Message::user("What is 8 * 7? Think step by step.");

        let result = model
            .chat_stream(Request::new("deepseek-reasoner", vec![msg]).with_stream(true))
            .await;
        assert!(result.is_ok());

        let mut stream = result.unwrap();
        let mut full_content = String::new();
        let mut full_reasoning = String::new();

        while let Some(chunk) = stream.next().await {
            if !chunk.content.is_empty() {
                print!("{}", chunk.content);
                full_content.push_str(&chunk.content);
            }
            if !chunk.reasoning_content.is_empty() {
                full_reasoning.push_str(&chunk.reasoning_content);
            }
            if chunk.is_finished {
                println!("\nFinish reason: {:?}", chunk.finish_reason);
            }
        }

        assert!(!full_content.is_empty());
        // 推理模型应该返回推理内容
        if !full_reasoning.is_empty() {
            println!("\nReasoning content length: {}", full_reasoning.len());
        }
    }

    #[tokio::test]
    async fn test_chat_with_openrouter_gemini() {
        dotenv::dotenv().ok();

        let provider = match openrouter_provider_from_env() {
            Ok(p) => Arc::new(p),
            Err(_) => {
                eprintln!("OPENROUTER_API_KEY not set, skipping test");
                return;
            }
        };

        let mut model = ModelRouter::new();
        model.add_model_provider(
            "google/gemini-3-pro-preview",
            provider,
            &[ModelCapability::Chat],
        );

        let msg = Message::user("Say 'Hello, world!' in one sentence.");

        let result = model
            .chat(Request::new("google/gemini-3-pro-preview", vec![msg]))
            .await;
        assert!(result.is_ok());

        let message = result.unwrap();
        if let Message::Assistant { content, .. } = message {
            println!("Response: {:?}", content);
            assert!(!content.is_empty());
        } else {
            panic!("Expected Assistant message");
        }
    }

    #[tokio::test]
    async fn test_chat_stream_with_openrouter_gemini() {
        dotenv::dotenv().ok();

        let provider = match openrouter_provider_from_env() {
            Ok(p) => Arc::new(p),
            Err(_) => {
                eprintln!("OPENROUTER_API_KEY not set, skipping test");
                return;
            }
        };

        let mut model = ModelRouter::new();
        model.add_model_provider(
            "google/gemini-3-pro-preview",
            provider,
            &[ModelCapability::Chat],
        );

        let msg = Message::user("Count from 1 to 3, each number on a new line.");

        let result = model
            .chat_stream(Request::new("google/gemini-3-pro-preview", vec![msg]).with_stream(true))
            .await;
        assert!(result.is_ok());

        let mut stream = result.unwrap();
        let mut full_content = String::new();

        while let Some(chunk) = stream.next().await {
            print!("{}", chunk.content);
            full_content.push_str(&chunk.content);
            if chunk.is_finished {
                println!("\nFinish reason: {:?}", chunk.finish_reason);
            }
        }

        assert!(!full_content.is_empty());
    }

    #[test]
    fn test_model_provider_mapping() {
        let ds_provider = Arc::new(deepseek_provider("dummy_key"));
        let or_provider = Arc::new(openrouter_provider("dummy_key"));

        let mut model = ModelRouter::new();

        model.add_models_for_provider(
            &["deepseek-chat", "deepseek-reasoner"],
            ds_provider,
            &[ModelCapability::Chat],
        );
        model.add_model_provider(
            "google/gemini-3-pro-preview",
            or_provider,
            &[ModelCapability::Chat],
        );

        assert!(model.supports("deepseek-chat", ModelCapability::Chat));
        assert!(model.supports("deepseek-reasoner", ModelCapability::Chat));
        assert!(model.supports("google/gemini-3-pro-preview", ModelCapability::Chat));
    }

    #[test]
    fn request_selects_the_model() {
        let provider = Arc::new(deepseek_provider("dummy_key"));
        let mut model = ModelRouter::new();
        model.add_model_provider("deepseek-chat", provider, &[ModelCapability::Chat]);

        assert!(model.route("deepseek-chat", ModelCapability::Chat).is_ok());
        assert!(matches!(
            model.route("non-existent-model", ModelCapability::Chat),
            Err(RouterError::ModelNotFound(_))
        ));
    }

    #[test]
    fn reasoning_options_belong_to_the_request() {
        let request = Request::new("model", Vec::new())
            .with_response_format_json()
            .with_reasoning_effort("high")
            .with_thinking(true);

        assert!(request.extra.contains_key("response_format"));
        assert_eq!(request.extra.get("reasoning_effort"), Some(&json!("high")));
        assert_eq!(
            request.extra.get("thinking"),
            Some(&json!({ "type": "enabled" }))
        );
    }
}
