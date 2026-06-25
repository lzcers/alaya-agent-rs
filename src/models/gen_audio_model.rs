use std::{collections::HashMap, sync::Arc};

use crate::{
    core::Message,
    models::{ChatError, GenAudioCapability, GenAudioResponse},
    providers::{Provider, Request},
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Map, Value, json};

pub struct GenAudioModel {
    model_providers: HashMap<String, Arc<dyn Provider>>,
    active_model: Option<String>,
    audio_format: String,
    voice: Option<String>,
}

impl Default for GenAudioModel {
    fn default() -> Self {
        Self::new()
    }
}

impl GenAudioModel {
    pub fn new() -> Self {
        Self {
            model_providers: HashMap::new(),
            active_model: None,
            audio_format: "wav".to_string(),
            voice: None,
        }
    }

    pub fn add_model_provider(&mut self, model_name: &str, provider: Arc<dyn Provider>) {
        self.model_providers
            .entry(model_name.to_owned())
            .or_insert(provider);
    }

    pub fn add_models_for_provider(&mut self, model_names: &[&str], provider: Arc<dyn Provider>) {
        for model_name in model_names {
            self.add_model_provider(model_name, provider.clone());
        }
    }

    pub fn set_active_model(&mut self, model_name: &str) -> Result<(), ChatError> {
        if !self.model_providers.contains_key(model_name) {
            return Err(ChatError::ModelNotFound(model_name.to_owned()));
        }
        self.active_model = Some(model_name.to_owned());
        Ok(())
    }

    pub fn with_audio_format(mut self, audio_format: String) -> Self {
        self.audio_format = audio_format;
        self
    }

    pub fn with_voice(mut self, voice: String) -> Self {
        self.voice = Some(voice);
        self
    }

    fn get_provider(&self, model_name: &str) -> Result<&Arc<dyn Provider>, ChatError> {
        self.model_providers
            .get(model_name)
            .ok_or_else(|| ChatError::ModelNotFound(model_name.to_owned()))
    }

    pub fn active_model(&self) -> Option<&str> {
        self.active_model.as_deref()
    }

    fn audio_config(&self) -> Value {
        let mut audio = Map::new();
        audio.insert("format".to_string(), json!(self.audio_format));

        if let Some(voice) = &self.voice {
            audio.insert("voice".to_string(), json!(voice));
        }

        Value::Object(audio)
    }
}

#[async_trait]
impl GenAudioCapability for GenAudioModel {
    async fn gen_audio(&self, msgs: Vec<Message>) -> Result<GenAudioResponse, ChatError> {
        let model_name = self
            .active_model
            .as_ref()
            .ok_or_else(|| ChatError::ModelNotFound("No active model set".to_string()))?;

        let provider = self.get_provider(model_name)?;

        let mut extra = std::collections::HashMap::new();
        extra.insert("modalities".to_string(), json!(["text", "audio"]));
        extra.insert("audio".to_string(), self.audio_config());

        let mut request = Request::new(model_name, msgs).with_stream(true);
        request.extra = extra;

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
            return Err(ChatError::NoResponse);
        }

        Ok(GenAudioResponse {
            audio_data,
            transcript,
            format: self.audio_format.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::{
        ChoiceAudio, Delta, ProviderError, StreamChoice, StreamResponse, openrouter_provider,
        openrouter_provider_from_env,
    };
    use futures::stream::{self, BoxStream};

    struct MockAudioProvider;

    #[async_trait]
    impl Provider for MockAudioProvider {
        async fn chat(
            &self,
            _request: Request,
        ) -> Result<crate::providers::Response, ProviderError> {
            Err(ProviderError::ApiError {
                code: 400,
                message: "chat is not used by GenAudioModel".to_string(),
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
        let mut model = GenAudioModel::new();
        model.add_model_provider("google/lyria-3-clip-preview", provider);
        model
            .set_active_model("google/lyria-3-clip-preview")
            .unwrap();

        let response = model
            .gen_audio(vec![Message::user("Generate a short piano loop")])
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

        let mut model = GenAudioModel::new().with_audio_format("wav".to_string());
        model.add_model_provider("google/lyria-3-clip-preview", provider);

        if let Err(e) = model.set_active_model("google/lyria-3-clip-preview") {
            eprintln!("Failed to set active model: {}", e);
            return;
        }

        let prompt = std::env::var("LYRIA_TEST_PROMPT")
            .unwrap_or_else(|_| "Generate a short upbeat synth loop with no vocals".to_string());
        let msg = Message::user(prompt);

        let result = model.gen_audio(vec![msg]).await;
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

        let mut model = GenAudioModel::new();

        model.add_models_for_provider(
            &["google/lyria-3-clip-preview", "google/lyria-3-pro-preview"],
            or_provider,
        );

        assert!(
            model
                .model_providers
                .contains_key("google/lyria-3-clip-preview")
        );
        assert!(
            model
                .model_providers
                .contains_key("google/lyria-3-pro-preview")
        );
        assert_eq!(model.model_providers.len(), 2);
    }

    #[test]
    fn test_set_active_model() {
        let provider = Arc::new(openrouter_provider("dummy_key"));
        let mut model = GenAudioModel::new();
        model.add_model_provider("google/lyria-3-clip-preview", provider);

        let result = model.set_active_model("google/lyria-3-clip-preview");
        assert!(result.is_ok());
        assert_eq!(
            model.active_model,
            Some("google/lyria-3-clip-preview".to_string())
        );

        let result = model.set_active_model("non-existent-model");
        assert!(result.is_err());
        assert!(matches!(result, Err(ChatError::ModelNotFound(_))));
    }

    #[test]
    fn test_builder_methods() {
        let model = GenAudioModel::new()
            .with_audio_format("mp3".to_string())
            .with_voice("alloy".to_string());

        assert_eq!(model.audio_format, "mp3");
        assert_eq!(model.voice, Some("alloy".to_string()));
        assert_eq!(
            model.audio_config(),
            json!({ "format": "mp3", "voice": "alloy" })
        );
    }
}
