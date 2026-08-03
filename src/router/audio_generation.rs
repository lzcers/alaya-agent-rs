use crate::{
    providers::Request,
    router::{ModelCapability, ModelRouter, RouterError},
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenAudioResponse {
    pub audio_data: String,
    pub transcript: String,
    pub format: String,
}

#[async_trait]
pub trait GenAudioCapability {
    async fn gen_audio(&self, request: Request) -> Result<GenAudioResponse, RouterError>;
}

#[async_trait]
impl GenAudioCapability for ModelRouter {
    async fn gen_audio(&self, request: Request) -> Result<GenAudioResponse, RouterError> {
        let provider = self.route(&request.model, ModelCapability::Audio)?;
        let format = audio_format(&request);
        let mut stream = provider.chat_stream(request).await?;
        let mut audio_data = String::new();
        let mut transcript = String::new();

        while let Some(response) = stream.next().await {
            for choice in response.choices {
                if let Some(audio) = choice.delta.audio {
                    if let Some(data) = audio.data {
                        audio_data.push_str(&data);
                    }
                    if let Some(chunk) = audio.transcript {
                        transcript.push_str(&chunk);
                    }
                }
            }
        }

        if audio_data.is_empty() {
            return Err(RouterError::NoResponse);
        }

        Ok(GenAudioResponse {
            audio_data,
            transcript,
            format,
        })
    }
}

fn audio_format(request: &Request) -> String {
    request
        .extra
        .get("audio")
        .and_then(Value::as_object)
        .and_then(|audio| audio.get("format"))
        .and_then(Value::as_str)
        .unwrap_or("wav")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;
    use crate::providers::{
        ChoiceAudio, Delta, Provider, ProviderError, StreamChoice, StreamResponse,
        openrouter_provider, openrouter_provider_from_env,
    };
    use futures::stream::{self, BoxStream};
    use serde_json::json;
    use std::sync::Arc;

    struct MockAudioProvider;

    #[async_trait]
    impl Provider for MockAudioProvider {
        async fn chat(
            &self,
            _request: Request,
        ) -> Result<crate::providers::Response, ProviderError> {
            Err(ProviderError::ApiError {
                code: 400,
                message: "chat is not used by the audio capability".to_string(),
            })
        }

        async fn chat_stream(
            &self,
            request: Request,
        ) -> Result<BoxStream<'static, StreamResponse>, ProviderError> {
            assert_eq!(
                request.extra.get("modalities"),
                Some(&json!(["text", "audio"]))
            );
            assert_eq!(
                request.extra.get("audio"),
                Some(&json!({ "format": "wav" }))
            );

            Ok(Box::pin(stream::iter(vec![
                audio_chunk("abc", "hello "),
                audio_chunk("def", "world"),
            ])))
        }

        fn name(&self) -> &str {
            "mock-audio"
        }
    }

    fn audio_chunk(data: &str, transcript: &str) -> StreamResponse {
        StreamResponse {
            id: "chunk_123".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 1743916800,
            model: "google/lyria-3-clip-preview".to_string(),
            system_fingerprint: None,
            choices: vec![StreamChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    reasoning_content: None,
                    audio: Some(ChoiceAudio {
                        data: Some(data.to_string()),
                        transcript: Some(transcript.to_string()),
                    }),
                    tool_calls: None,
                },
                finish_reason: None,
                logprobs: None,
            }],
            usage: None,
        }
    }

    #[tokio::test]
    async fn test_gen_audio_collects_streamed_audio_chunks() {
        let provider = Arc::new(MockAudioProvider);
        let mut model = ModelRouter::new();
        model.add_model_provider(
            "google/lyria-3-clip-preview",
            provider,
            &[ModelCapability::Audio],
        );
        let response = model
            .gen_audio(
                Request::new(
                    "google/lyria-3-clip-preview",
                    vec![Message::user("Generate a short piano loop")],
                )
                .with_stream(true)
                .with_audio("wav", None),
            )
            .await
            .unwrap();

        assert_eq!(response.audio_data, "abcdef");
        assert_eq!(response.transcript, "hello world");
        assert_eq!(response.format, "wav");
    }

    #[tokio::test]
    #[ignore = "uses paid OpenRouter audio generation"]
    async fn test_gen_audio_with_openrouter() {
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
            "google/lyria-3-clip-preview",
            provider,
            &[ModelCapability::Audio],
        );

        let prompt = std::env::var("LYRIA_TEST_PROMPT")
            .unwrap_or_else(|_| "Generate a short upbeat synth loop with no vocals".to_string());
        let msg = Message::user(prompt);

        let result = model
            .gen_audio(
                Request::new("google/lyria-3-clip-preview", vec![msg])
                    .with_stream(true)
                    .with_audio("wav", None),
            )
            .await;
        if let Err(e) = result {
            eprintln!("Failed to generate audio: {}", e);
            return;
        }

        assert!(result.is_ok());

        let response = result.unwrap();
        assert!(!response.audio_data.is_empty());
        assert_eq!(response.format, "wav");

        if let Ok(output_path) = std::env::var("LYRIA_TEST_OUTPUT_B64") {
            std::fs::write(output_path, &response.audio_data)
                .expect("failed to write generated audio base64");
        }
    }

    #[test]
    fn test_model_provider_mapping() {
        let or_provider = Arc::new(openrouter_provider("dummy_key"));

        let mut model = ModelRouter::new();

        model.add_models_for_provider(
            &["google/lyria-3-clip-preview", "google/lyria-3-pro-preview"],
            or_provider,
            &[ModelCapability::Audio],
        );

        assert!(model.supports("google/lyria-3-clip-preview", ModelCapability::Audio));
        assert!(model.supports("google/lyria-3-pro-preview", ModelCapability::Audio));
    }

    #[test]
    fn request_selects_the_model() {
        let provider = Arc::new(openrouter_provider("dummy_key"));
        let mut model = ModelRouter::new();
        model.add_model_provider(
            "google/lyria-3-clip-preview",
            provider,
            &[ModelCapability::Audio],
        );

        assert!(
            model
                .route("google/lyria-3-clip-preview", ModelCapability::Audio)
                .is_ok()
        );
        assert!(matches!(
            model.route("non-existent-model", ModelCapability::Audio),
            Err(RouterError::ModelNotFound(_))
        ));
    }

    #[test]
    fn audio_options_belong_to_the_request() {
        let request = Request::new("model", Vec::new())
            .with_stream(true)
            .with_audio("mp3", Some("alloy".to_string()));

        assert_eq!(audio_format(&request), "mp3");
        assert_eq!(
            request.extra.get("audio"),
            Some(&json!({ "format": "mp3", "voice": "alloy" }))
        );
    }
}
