//! Chat client implementation for `/chat/completions`.
use crate::DeepSeekRequest;
use crate::error::DeepSeekError;
use crate::{api_post, api_request_stream};

use super::{Chat, ChatStream, request::*};
use futures_util::StreamExt;
use reqwest::Method;
use reqwest_eventsource::Event;
use std::sync::mpsc as std_mpsc;
use tokio::sync::mpsc;
/// Stream item produced by chat streaming.
pub type ChatStreamItem = Result<ChatStream, DeepSeekError>;

/// Blocking iterator over streaming chat chunks.
pub struct ChatStreamBlocking {
    pub rx: std_mpsc::Receiver<ChatStreamItem>,
}

impl Iterator for ChatStreamBlocking {
    type Item = ChatStreamItem;

    fn next(&mut self) -> Option<Self::Item> {
        self.rx.recv().ok()
    }
}

impl DeepSeekRequest for ChatRequest {
    type Response = Chat;
    type StreamItem = ChatStreamItem;
    type BlockingStream = ChatStreamBlocking;

    async fn send(self) -> Result<Chat, DeepSeekError> {
        let credentials = self.credentials.clone();
        api_post("/chat/completions", &self, credentials).await
    }

    async fn stream(self) -> Result<mpsc::Receiver<ChatStreamItem>, DeepSeekError> {
        let mut request = self;
        request.stream = Some(true);

        let credentials = request.credentials.clone();
        let mut event_source = api_request_stream(
            Method::POST,
            "/chat/completions",
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
                        match serde_json::from_str::<ChatStream>(&message.data) {
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

    fn stream_blocking(self) -> Result<ChatStreamBlocking, DeepSeekError> {
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

        Ok(ChatStreamBlocking { rx })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Credentials, DEFAULT_BASE_URL};

    fn get_credentials() -> Credentials {
        Credentials::new(
            std::env::var("DEEPSEEK_API").unwrap(),
            DEFAULT_BASE_URL.clone(),
        )
    }

    fn get_builder() -> ChatRequestBuilder {
        ChatRequestBuilder::default()
            .credentials(get_credentials())
            .model("deepseek-v4-flash")
            .thinking(Thinking::disabled())
    }

    #[tokio::test]
    async fn chat() {
        let req = get_builder()
            .message(ChatMessage::User {
                content: "Hi".to_string(),
                name: None,
            })
            .max_tokens(5 as u32)
            .logprobs(true)
            .top_logprobs(2 as u32)
            .build()
            .unwrap();
        let response = req.send().await.unwrap();
        println!("{:#?}", response);
    }

    #[tokio::test]
    async fn api_error() {
        let mut req = get_builder()
            .message(ChatMessage::User {
                content: "Hi".to_string(),
                name: None,
            })
            .build()
            .unwrap();
        req.reasoning_effort = Some(ReasoningEffort::Max);
        let response = req.send().await;
        assert!(response.is_err());
        if let Err(err) = response {
            assert!(matches!(err, DeepSeekError::Api { .. }));
            if let DeepSeekError::Api {
                error,
                status,
                body,
            } = err
            {
                assert_eq!(status, Some(400));
                assert!(body.is_some());
                assert_eq!(
                    error.message,
                    "thinking options type cannot be disabled when reasoning_effort is set"
                );
                assert_eq!(error.error_type, "invalid_request_error");
                assert_eq!(error.param.as_deref(), None);
                assert_eq!(error.code.as_deref(), Some("invalid_request_error"));
            } else {
                panic!("Expected DeepSeekError::Api");
            }
        }
    }

    #[tokio::test]
    async fn chat_tool_call() {
        let mut messages = vec![ChatMessage::User {
            content: "How's the weather in Hangzhou, Zhejiang?".to_string(),
            name: None,
        }];
        let req_tool = Tool::new(
            "get_weather",
            "Get weather of a location, the user should supply a location first.",
            Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                },
                "required": ["location"]
            })),
        );
        let req = get_builder()
            .tool(req_tool.clone())
            .messages(messages.clone())
            .build()
            .unwrap();
        let message = req.send().await.unwrap().choices[0].clone().message;
        let Some(tool_calls) = message.tool_calls.clone() else {
            return;
        };
        let tool_call = tool_calls[0].clone();
        messages.push(ChatMessage::Assistant {
            content: message.content,
            name: None,
            tool_calls: Some(tool_calls),
        });
        messages.push(ChatMessage::Tool {
            tool_call_id: tool_call.id,
            content: "24°C".to_string(),
        });

        let req2 = get_builder()
            .tool(req_tool)
            .messages(messages)
            .build()
            .unwrap();
        let response = req2.send().await.unwrap();
        println!("{:#?}", response);
        assert!(
            response.choices[0]
                .message
                .content
                .as_ref()
                .unwrap()
                .contains("24°C")
        );
    }

    #[tokio::test]
    async fn chat_stream_async() {
        let req = get_builder()
            .message(ChatMessage::User {
                content: "Hi".to_string(),
                name: None,
            })
            .max_tokens(16 as u32)
            .build()
            .unwrap();

        let mut rx = req.stream().await.unwrap();
        while let Some(item) = rx.recv().await {
            match item {
                Ok(chunk) => println!("Model>\t {:#?}", chunk),
                Err(err) => eprintln!("Error>\t {:#?}", err),
            }
        }
    }

    #[test]
    fn chat_stream_blocking() {
        let req = get_builder()
            .message(ChatMessage::User {
                content: "Hi".to_string(),
                name: None,
            })
            .max_tokens(16 as u32)
            .build()
            .unwrap();

        let mut stream = req.stream_blocking().unwrap();
        let mut content = String::new();

        for item in stream.by_ref().take(50) {
            let chunk = item.unwrap();
            for choice in chunk.choices {
                if let Some(delta_content) = choice.delta.content {
                    content.push_str(&delta_content);
                }
            }
        }

        println!("Model>\t {}", content);
        assert!(!content.is_empty());
    }
}
