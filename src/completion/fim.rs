//! FIM (Fill-In-the-Middle) completion models and request types.
//!
//! This endpoint is beta and requires the beta base URL:
//! `https://api.deepseek.com/beta`.
use std::collections::HashMap;
use std::sync::mpsc as std_mpsc;

use crate::chat::request::{Stop, StreamOptions, is_none_or_empty_stop};
use crate::chat::response::ChatGeneric;
use crate::error::DeepSeekError;
use crate::{Credentials, api_request_stream};
use crate::{DeepSeekRequest, api_post};
use derive_builder::Builder;
use futures_util::StreamExt;
use reqwest::Method;
use reqwest_eventsource::Event;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// Non-streaming FIM completion response.
pub type Completion = ChatGeneric<CompletionChoice>;

/// FIM completion request payload.
#[derive(Clone, Debug, PartialEq, Serialize, Builder)]
#[builder(
    pattern = "owned",
    setter(into, strip_option),
    build_fn(validate = "Self::validate"),
    name = "FIMCompletionRequestBuilder"
)]
pub struct FIMCompletionRequest {
    #[serde(skip_serializing)]
    #[builder(default)]
    pub credentials: Option<Credentials>,

    pub model: String,
    pub prompt: String,
    #[builder(default)]
    pub echo: Option<bool>,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// string | string[] | null
    #[builder(default)]
    #[serde(skip_serializing_if = "is_none_or_empty_stop")]
    pub stop: Option<Stop>,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,

    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,

    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    #[builder(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
}

impl FIMCompletionRequestBuilder {
    fn validate(&self) -> Result<(), String> {
        if let Some(temperature) = self.temperature.flatten() {
            if !(0.0..=2.0).contains(&temperature) {
                return Err("temperature must be between 0 and 2".to_string());
            }
        }
        if let Some(logprobs) = self.logprobs.flatten() {
            if logprobs > 20 {
                return Err("logprobs must be <= 20".to_string());
            }
        }

        if let Some(top_p) = self.top_p.flatten() {
            if !(0.0..=1.0).contains(&top_p) {
                return Err("top_p must be between 0 and 1".to_string());
            }
        }

        if let Some(stream) = self.stream.flatten() {
            if !stream && self.stream_options.is_some() {
                return Err("stream_options cannot be set when stream is false".to_string());
            }
        }
        Ok(())
    }
}

/// FIM completion choice.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct CompletionChoice {
    pub finish_reason: FinishReason,
    pub index: u64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}

/// Completion finish reason.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ContentFilter,
    InsufficientSystemResources,
}

/// Logprob details for completion tokens.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Logprobs {
    pub text_offset: Vec<u64>,
    pub token_logprobs: Vec<f64>,
    pub tokens: Vec<String>,
    pub top_logprobs: Option<Vec<HashMap<String, f64>>>,
}
/// Streaming completion choice.
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct CompletionChoiceStream {
    pub finish_reason: Option<FinishReason>,
    pub index: u64,
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Logprobs>,
}
/// ```text
/// data: {"id":"fb50cff8-93f0-49ee-b6c7-2878bae940fa","choices":[{"text":"","index":0,"logprobs":null,"finish_reason":null}],"created":1779503544,"model":"deepseek-v4-flash","system_fingerprint":"fp_8b330d02d0_prod0820_fp8_kvcache_20260402","object":"text_completion"}
/// data: {"id":"fb50cff8-93f0-49ee-b6c7-2878bae940fa","choices":[{"text":"18","index":0,"logprobs":{"tokens":["18"],"token_logprobs":[-3.5918827],"top_logprobs":[{"20":-2.850668,"3":-2.7995281}],"text_offset":[18]},"finish_reason":null}],"created":1779503544,"model":"deepseek-v4-flash","system_fingerprint":"fp_8b330d02d0_prod0820_fp8_kvcache_20260402","object":"text_completion"}
/// data: {"id":"fb50cff8-93f0-49ee-b6c7-2878bae940fa","choices":[{"text":"-year","index":0,"logprobs":{"tokens":["-year"],"token_logprobs":[-0.95153236],"top_logprobs":[{"-year":-0.95153236," years":-1.1679096}],"text_offset":[20]},"finish_reason":null}],"created":1779503544,"model":"deepseek-v4-flash","system_fingerprint":"fp_8b330d02d0_prod0820_fp8_kvcache_20260402","object":"text_completion"}
/// data: {"id":"fb50cff8-93f0-49ee-b6c7-2878bae940fa","choices":[{"text":"-old","index":0,"logprobs":{"tokens":["-old"],"token_logprobs":[-0.046930313],"top_logprobs":[{" old":-3.628458,"-old":-0.046930313}],"text_offset":[25]},"finish_reason":null}],"created":1779503544,"model":"deepseek-v4-flash","system_fingerprint":"fp_8b330d02d0_prod0820_fp8_kvcache_20260402","object":"text_completion"}
/// ...
/// data: [DONE]
/// ```

/// Streaming FIM completion response (SSE chunks).
pub type CompletionStream = ChatGeneric<CompletionChoiceStream>;
/// Stream item produced by FIM completion streaming.
pub type CompletionStreamItem = Result<CompletionStream, DeepSeekError>;
/// Blocking iterator over FIM completion streaming chunks.
pub struct CompletionStreamBlocking {
    rx: std_mpsc::Receiver<CompletionStreamItem>,
}

impl Iterator for CompletionStreamBlocking {
    type Item = CompletionStreamItem;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}
impl DeepSeekRequest for FIMCompletionRequest {
    type Response = Completion;
    type StreamItem = CompletionStreamItem;
    type BlockingStream = CompletionStreamBlocking;

    async fn send(self) -> Result<Self::Response, DeepSeekError> {
        let credentials = self.credentials.clone();
        api_post("/completions", &self, credentials).await
    }

    async fn stream(self) -> Result<mpsc::Receiver<Self::StreamItem>, DeepSeekError> {
        let mut request = self;
        request.stream = Some(true);

        let credentials = request.credentials.clone();
        let mut event_source = api_request_stream(
            Method::POST,
            "/completions",
            |builder| builder.json(&request),
            credentials,
        )
        .await?;

        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            while let Some(event) = event_source.next().await {
                match event {
                    Ok(Event::Open) => {}
                    Ok(Event::Message(message)) => {
                        if message.data == "[DONE]" {
                            break;
                        }
                        match serde_json::from_str::<CompletionStream>(&message.data) {
                            Ok(chunk) => {
                                if tx.send(Ok(chunk)).await.is_err() {
                                    break;
                                }
                            }
                            Err(err) => {
                                let _ = tx
                                    .send(Err(DeepSeekError::decode(err.to_string(), message.data)))
                                    .await;
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        let _ = tx
                            .send(Err(DeepSeekError::decode(err.to_string(), String::new())))
                            .await;
                        break;
                    }
                }
            }
        });

        Ok(rx)
    }

    fn stream_blocking(self) -> Result<CompletionStreamBlocking, DeepSeekError> {
        let (tx, rx) = std_mpsc::channel();

        std::thread::spawn(move || {
            let runtime = match tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
            {
                Ok(runtime) => runtime,
                Err(err) => {
                    let _ = tx.send(Err(DeepSeekError::decode(err.to_string(), String::new())));
                    return;
                }
            };

            runtime.block_on(async move {
                match self.stream().await {
                    Ok(mut stream_rx) => {
                        while let Some(item) = stream_rx.recv().await {
                            if tx.send(item).is_err() {
                                break;
                            }
                        }
                    }
                    Err(err) => {
                        let _ = tx.send(Err(err));
                    }
                }
            });
        });

        Ok(CompletionStreamBlocking { rx })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Credentials;
    use crate::DEFAULT_BETA_BASE_URL;

    fn get_credentials() -> Credentials {
        Credentials::new(
            std::env::var("DEEPSEEK_API").expect("DEEPSEEK_API is not set"),
            DEFAULT_BETA_BASE_URL.clone(),
        )
    }

    fn get_fim_builder() -> FIMCompletionRequestBuilder {
        FIMCompletionRequestBuilder::default()
            .credentials(get_credentials())
            .model("deepseek-v4-flash")
            .max_tokens(64 as u32)
    }

    #[tokio::test]
    async fn test_fim_completion() {
        let fim_request = get_fim_builder()
            .prompt("def fib(a):")
            .suffix("    return fib(a-1) + fib(a-2)")
            .build()
            .unwrap();
        let response = fim_request.send().await.unwrap();
        println!("{:#?}", response);
        assert_eq!(response.object, "text_completion");
        assert_eq!(response.model, "deepseek-v4-flash");
        assert_eq!(response.choices.len(), 1);
    }

    #[tokio::test]
    async fn test_fim_completion_stream() {
        let fim_request = get_fim_builder()
            .prompt("def fib(a):")
            .suffix("    return fib(a-1) + fib(a-2)")
            .stream(true)
            .build()
            .unwrap();
        let mut stream = fim_request.stream().await.unwrap();
        while let Some(item) = stream.recv().await {
            match item {
                Ok(chunk) => println!("Received chunk: {:#?}", chunk),
                Err(err) => eprintln!("Stream error: {}", err),
            }
        }
    }

    #[tokio::test]
    async fn test_fim_completion_stream_blocking() {
        let fim_request = get_fim_builder()
            .prompt("def fib(a):")
            .suffix("    return fib(a-1) + fib(a-2)")
            .stream(true)
            .build()
            .unwrap();
        let mut stream = fim_request.stream_blocking().unwrap();
        while let Some(item) = stream.next() {
            match item {
                Ok(chunk) => println!("Received chunk: {:#?}", chunk),
                Err(err) => eprintln!("Stream error: {}", err),
            }
        }
    }
}
