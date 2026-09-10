use async_trait::async_trait;

use crate::capability::ModelError;

/// 一次音频生成的域请求。
///
/// 替代了此前「构造一个带 `modalities` / `audio` 字段的 chat wire 请求」的写法：
/// 这些字段属于协议层，不属于调用方。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenAudioRequest {
    pub model: String,
    pub prompt: String,
    /// 输出格式，缺省由协议层决定。
    pub format: Option<String>,
    pub voice: Option<String>,
}

impl GenAudioRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            format: None,
            voice: None,
        }
    }

    pub fn with_format(mut self, format: impl Into<String>) -> Self {
        self.format = Some(format.into());
        self
    }

    pub fn with_voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }
}

/// 音频生成结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenAudioResponse {
    /// base64 编码的音频数据
    pub audio_data: String,
    pub transcript: String,
    pub format: String,
}

/// 音频生成能力。
#[async_trait]
pub trait GenAudioCapability: Send + Sync {
    async fn gen_audio(&self, request: GenAudioRequest) -> Result<GenAudioResponse, ModelError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_options_belong_to_the_request() {
        let request = GenAudioRequest::new("model", "prompt")
            .with_format("mp3")
            .with_voice("alloy");

        assert_eq!(request.format.as_deref(), Some("mp3"));
        assert_eq!(request.voice.as_deref(), Some("alloy"));
    }
}
