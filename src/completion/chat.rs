//! Beta chat completion request types.
//!
//! The beta endpoint enables prefix/continuation behavior for assistant messages.
pub mod request {
    use crate::DeepSeekError;
    use crate::chat::request::{
        ReasoningEffort, ResponseFormat, Stop, StreamOptions, Thinking, ThinkingType, ToolChoice,
        ToolType, is_none_or_empty_stop,
    };
    use crate::chat::response::ToolCall;
    use crate::chat::{Chat, ChatStream, ChatStreamBlocking, ChatStreamItem, is_none_or_empty_vec};
    use crate::{Credentials, DeepSeekRequest, api_post, api_request_stream};
    use derive_builder::Builder;
    use futures_util::StreamExt;
    use reqwest::Method;
    use reqwest_eventsource::Event;
    use serde::{Deserialize, Serialize};
    use std::sync::mpsc as std_mpsc;
    use tokio::sync::mpsc;

    fn is_false(value: &bool) -> bool {
        !*value
    }

    /// Beta chat request payload (beta base URL required).
    #[derive(Clone, Debug, PartialEq, Serialize, Builder)]
    #[builder(
        pattern = "owned",
        setter(into, strip_option),
        build_fn(validate = "Self::validate"),
        name = "BetaChatRequestBuilder"
    )]
    pub struct BetaChatRequest {
        #[serde(skip_serializing)]
        #[builder(default)]
        pub credentials: Option<Credentials>,

        #[builder(setter(each(name = "message", into)))]
        pub messages: Vec<BetaChatMessage>,
        pub model: String,
        /// 推理开关对象：{"type": "enabled" | "disabled"}。
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub thinking: Option<Thinking>,
        /// 控制推理强度（high / max）。
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reasoning_effort: Option<ReasoningEffort>,
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tokens: Option<u32>,
        /// Must be one of text or json_object.
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_format: Option<ResponseFormat>,

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

        /// 采样温度，介于 0 和 2 之间。更高的值，如 0.8，会使输出更随机，而更低的值，如 0.2，会使其更加集中和确定。
        /// 我们通常建议可以更改这个值或者更改 top_p，但不建议同时对两者进行修改。
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f64>,

        /// 作为调节采样温度的替代方案，模型会考虑前 top_p 概率的 token 的结果。所以 0.1 就意味着只有包括在最高 10% 概率中的 token 会被考虑。
        /// 我们通常建议修改这个值或者更改 temperature，但不建议同时对两者进行修改。
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f64>,

        #[builder(default, setter(each(name = "tool", into)))]
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub tools: Vec<Tool>,

        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_choice: Option<ToolChoice>,

        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<bool>,

        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_logprobs: Option<u32>,

        /// 您自定义的 user_id，可选字符集为 [a-zA-Z0-9\-_]，最大长度为 512。请不要在 user_id 中包含用户隐私信息。
        /// user_id 可用于区分您业务侧的用户身份，以帮助我们进行内容安全处理。
        /// user_id 可用于 KVCache 缓存隔离，以进行隐私管理。
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user_id: Option<String>,
    }
    /// Beta chat message variants.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(tag = "role", rename_all = "snake_case")]
    pub enum BetaChatMessage {
        System {
            content: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<String>,
        },
        User {
            content: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<String>,
        },
        Assistant {
            #[serde(skip_serializing_if = "Option::is_none")]
            content: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<String>,
            /// (Beta) 设置此参数为 true，来强制模型在其回答中以此 `assistant` 消息中提供的前缀内容开始。
            /// 您必须设置 `base_url="https://api.deepseek.com/beta"` 来使用此功能。
            #[serde(default, skip_serializing_if = "is_false")]
            prefix: bool,
            /// (Beta) 用于思考模式下在对话前缀续写功能下，作为最后一条 assistant 思维链内容的输入。
            /// 使用此功能时，prefix 参数必须设置为 true。
            #[serde(skip_serializing_if = "Option::is_none")]
            reasoning_content: Option<String>,
            #[serde(skip_serializing_if = "is_none_or_empty_vec")]
            tool_calls: Option<Vec<ToolCall>>,
        },
        Tool {
            content: String,
            tool_call_id: String,
        },
    }
    /// Tool definition for beta chat requests.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    pub struct Tool {
        #[serde(rename = "type")]
        pub typ: ToolType,
        pub function: BetaToolFunctionDefinition,
    }

    impl Tool {
        pub fn new(
            name: impl Into<String>,
            description: impl Into<String>,
            parameters: Option<serde_json::Value>,
            strict: Option<bool>,
        ) -> Self {
            Tool {
                typ: ToolType::Function,
                function: BetaToolFunctionDefinition {
                    name: name.into(),
                    description: description.into(),
                    parameters,
                    strict,
                },
            }
        }
    }
    /// Tool function definition for beta chat requests.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct BetaToolFunctionDefinition {
        pub description: String,
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub parameters: Option<serde_json::Value>,
        /// (Beta) 是否严格按照提供的参数 JSON Schema 定义进行校验和调用，默认为 false。
        pub strict: Option<bool>,
    }

    impl BetaChatRequestBuilder {
        fn validate(&self) -> Result<(), String> {
            // derive_builder + strip_option makes Option<T> fields become Option<Option<T>> here;
            // flatten() treats "unset" and "explicit None" uniformly for validation.
            if let Some(temperature) = self.temperature.flatten() {
                if !(0.0..=2.0).contains(&temperature) {
                    return Err("temperature must be between 0 and 2".to_string());
                }
            }

            if let Some(top_p) = self.top_p.flatten() {
                if !(0.0..=1.0).contains(&top_p) {
                    return Err("top_p must be between 0 and 1".to_string());
                }
            }

            if let Some(top_logprobs) = self.top_logprobs.flatten() {
                if top_logprobs > 20 {
                    return Err("top_logprobs must be <= 20".to_string());
                }
                if self.logprobs.flatten() != Some(true) {
                    return Err("top_logprobs requires logprobs=true".to_string());
                }
            }

            if let Some(thinking) = self
                .thinking
                .as_ref()
                .and_then(|thinking| thinking.as_ref())
            {
                if let Some(reasoning_effort) = self
                    .reasoning_effort
                    .as_ref()
                    .and_then(|effort| effort.as_ref())
                {
                    if matches!(thinking.typ, ThinkingType::Disabled)
                        && matches!(
                            reasoning_effort,
                            ReasoningEffort::High | ReasoningEffort::Max
                        )
                    {
                        return Err(
                            "thinking options type cannot be disabled when reasoning_effort is set"
                                .to_string(),
                        );
                    }
                }
            }

            if let Some(stream) = self.stream.flatten() {
                if !stream && self.stream_options.is_some() {
                    return Err("stream_options cannot be set when stream is false".to_string());
                }
            }

            if let Some(messages) = self.messages.as_ref() {
                messages.iter().try_for_each(|message| {
                    if let BetaChatMessage::Assistant {
                        prefix: false,
                        reasoning_content: Some(_),
                        ..
                    } = message
                    {
                        return Err(
                            "reasoning_content cannot be set when assistant message prefix is false".to_string(),
                        );
                    }
                    Ok(())
                })?;
            }

            Ok(())
        }
    }

    impl DeepSeekRequest for BetaChatRequest {
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
                                        .send(Err(DeepSeekError::decode(
                                            err.to_string(),
                                            message.data,
                                        )))
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
}

#[cfg(test)]
mod tests {
    use super::request::*;
    use crate::{Credentials, DEFAULT_BETA_BASE_URL, DeepSeekRequest, chat::request::Thinking};

    fn get_credentials() -> Credentials {
        Credentials::new(
            std::env::var("DEEPSEEK_API").unwrap(),
            DEFAULT_BETA_BASE_URL.clone(),
        )
    }

    fn get_builder() -> BetaChatRequestBuilder {
        BetaChatRequestBuilder::default()
            .credentials(get_credentials())
            .model("deepseek-v4-flash")
            .max_tokens(32 as u32)
            .thinking(Thinking::disabled())
    }

    #[tokio::test]
    async fn beta_chat() {
        let req = get_builder()
            .message(BetaChatMessage::User {
                content: "Please write quick sort code".to_string(),
                name: None,
            })
            .message(BetaChatMessage::Assistant {
                content: Some("```python\n".to_string()),
                name: None,
                prefix: true,
                reasoning_content: None,
                tool_calls: None,
            })
            .stop("```")
            .build()
            .unwrap();
        let response = req.send().await.unwrap();
        println!("{:#?}", response);
    }
}
