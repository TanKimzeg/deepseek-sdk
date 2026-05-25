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
    use crate::{DeepSeekClient, DeepSeekRequest, api_post, api_request_stream};
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
    #[derive(Clone, Debug, Serialize, Builder)]
    #[builder(
        pattern = "owned",
        setter(into, strip_option),
        build_fn(validate = "Self::validate"),
        name = "BetaChatRequestBuilder"
    )]
    pub struct BetaChatRequest {
        #[serde(skip_serializing)]
        pub client: DeepSeekClient,

        /// A list of messages comprising the conversation so far.
        #[builder(setter(each(name = "message", into)))]
        pub messages: Vec<BetaChatMessage>,

        /// Possible values: [`deepseek-v4-flash`, `deepseek-v4-pro`]
        ///
        /// ID of the model to use.
        pub model: String,
        /// 推理开关对象：{"type": "enabled" | "disabled"}。
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]

        /// Controls the switch between thinking and non-thinking mode.
        pub thinking: Option<Thinking>,

        /// Possible values: [`high`, `max`]
        ///
        /// Controls the reasoning effort of the model.
        /// The default effort is `high` for regular requests;
        /// for some complex agent requests (such as Claude Code, OpenCode),
        /// effort is automatically set to `max`.
        /// For compatibility, `low` and `medium` are mapped to `high`,
        /// and `xhigh` is mapped to `max`.
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reasoning_effort: Option<ReasoningEffort>,

        /// The maximum number of tokens that can be generated in the chat completion.
        ///
        /// The total length of input tokens and generated tokens is limited by the model's context length.
        ///
        /// For the value range and default value, please refer to the [documentation](https://api-docs.deepseek.com/quick_start/pricing).
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tokens: Option<u32>,

        /// An object specifying the format that the model must output.
        /// Setting to { "type": "json_object" } enables JSON Output,
        /// which guarantees the message the model generates is valid JSON.
        ///
        /// **Important**: When using JSON Output, you must also instruct the model to produce JSON yourself via a system or user message.
        /// Without this, the model may generate an unending stream of whitespace until the generation reaches the token limit, resulting in a long-running and seemingly "stuck" request. Also note that the message content may be partially cut off if finish_reason="length", which indicates the generation exceeded max_tokens or the conversation exceeded the max context length.
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub response_format: Option<ResponseFormat>,

        /// Up to 16 sequences where the API will stop generating further tokens.
        #[builder(default)]
        #[serde(skip_serializing_if = "is_none_or_empty_stop")]
        pub stop: Option<Stop>,

        /// If set, partial message deltas will be sent.
        /// Tokens will be sent as data-only server-sent events (SSE) as they become available,
        /// with the stream terminated by a `data: [DONE]`` message.
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,

        /// Options for streaming response. Only set this when you set `stream: true`.
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream_options: Option<StreamOptions>,

        /// Possible values: `<= 2`
        ///
        /// Default value: `1`
        ///
        /// What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        /// We generally recommend altering this or `top_p` but not both.
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f64>,

        /// Possible values: `<= 1`
        ///
        /// Default value: `1`
        ///
        /// An alternative to sampling with temperature, called nucleus sampling,
        /// where the model considers the results of the tokens with top_p probability mass.
        /// So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        ///
        /// We generally recommend altering this or `temperature` but not both.
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f64>,

        /// A list of tools the model may call. Currently, only functions are supported as a tool.
        /// Use this to provide a list of functions the model may generate JSON inputs for.
        /// A max of 128 functions are supported.
        #[builder(default, setter(each(name = "tool", into)))]
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub tools: Vec<Tool>,

        /// Controls which (if any) tool is called by the model.
        /// `none` means the model will not call any tool and instead generates a message.
        /// `auto` means the model can pick between generating a message or calling one or more tools.
        /// `required` means the model must call one or more tools.
        /// Specifying a particular tool via `{"type": "function", "function": {"name": "my_function"}}` forces the model to call that tool.
        /// `none` is the default when no tools are present. `auto` is the default if tools are present.
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub tool_choice: Option<ToolChoice>,

        /// Whether to return log probabilities of the output tokens or not.
        /// If true, returns the log probabilities of each output token returned in the `content` of `message`.
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<bool>,

        /// Possible values: `<= 20`
        ///
        /// An integer between 0 and 20 specifying the number of most likely tokens to return at each token position,
        /// each with an associated log probability. `logprobs` must be set to `true` if this parameter is used.
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_logprobs: Option<u32>,

        /// A custom `user_id`. Allowed character set is `[a-zA-Z0-9\-_]`, with a maximum length of 512.
        /// Do not include user privacy information in the `user_id`.

        /// `user_id` can be used to distinguish user identities on your side to help us with content safety review.
        /// `user_id` can be used for KVCache isolation for privacy management.
        /// `user_id` can be used for scheduling isolation of users on your business side.
        /// For more details on the `user_id` parameter, please refer to [Rate Limit & Isolation](https://api-docs.deepseek.com/quick_start/rate_limit)
        #[builder(default)]
        #[serde(skip_serializing_if = "Option::is_none")]
        pub user_id: Option<String>,
    }
    /// Beta chat message variants.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(tag = "role", rename_all = "snake_case")]
    pub enum BetaChatMessage {
        System {
            /// The contents of the system message.
            content: String,
            /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<String>,
        },
        User {
            /// The contents of the user message.
            content: String,
            /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<String>,
        },
        Assistant {
            /// The contents of the assistant message.
            #[serde(skip_serializing_if = "Option::is_none")]
            content: Option<String>,
            /// An optional name for the participant. Provides the model information to differentiate between participants of the same role.
            #[serde(skip_serializing_if = "Option::is_none")]
            name: Option<String>,
            /// (Beta) Set this to `true` to force the model to start its answer by the content of the supplied prefix in this `assistant` message.
            /// You must set `base_url="https://api.deepseek.com/beta"` to use this feature.
            #[serde(default, skip_serializing_if = "is_false")]
            prefix: bool,
            /// (Beta) Used for the thinking mode in the [Chat Prefix Completion](https://api-docs.deepseek.com/guides/chat_prefix_completion)
            /// feature as the input for the CoT in the last assistant message.
            /// When using this feature, the `prefix` parameter must be set to `true`.
            #[serde(skip_serializing_if = "Option::is_none")]
            reasoning_content: Option<String>,
            #[serde(skip_serializing_if = "is_none_or_empty_vec")]
            tool_calls: Option<Vec<ToolCall>>,
        },
        Tool {
            /// The contents of the tool message.
            content: String,
            /// Tool call that this message is responding to.
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
        /// (Beta) Default value: `false`
        ///
        /// If set to true, the API will use strict-mode for the tool calls to ensure the output always complies with the function's JSON schema.
        /// This is a Beta feature, for more details please refer to [Tool Calls Guide](https://api-docs.deepseek.com/zh-cn/guides/tool_calls)
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

            if let Some(stop) = self.stop.as_ref().and_then(|s| s.as_ref()) {
                if let Stop::Many(values) = stop {
                    if values.len() > 16 {
                        return Err("a maximum of 16 stop sequences are allowed".to_string());
                    }
                }
            }
            Ok(())
        }
    }

    impl DeepSeekRequest for BetaChatRequest {
        type Response = Chat;
        type StreamItem = ChatStreamItem;
        type BlockingStream = ChatStreamBlocking;

        async fn send(self) -> Result<Chat, DeepSeekError> {
            let client = self.client.clone();
            api_post("/chat/completions", &self, client).await
        }

        async fn stream(self) -> Result<mpsc::Receiver<ChatStreamItem>, DeepSeekError> {
            let mut request = self;
            request.stream = Some(true);

            let client = request.client.clone();
            let mut event_source = api_request_stream(
                Method::POST,
                "/chat/completions",
                |builder| builder.json(&request),
                client,
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
    use crate::{DEFAULT_BETA_BASE_URL, DeepSeekClient, DeepSeekRequest, chat::request::Thinking};

    fn get_client() -> DeepSeekClient {
        DeepSeekClient::new(
            std::env::var("DEEPSEEK_API").expect("DEEPSEEK_API is not set"),
            DEFAULT_BETA_BASE_URL.clone(),
        )
    }

    fn get_builder() -> BetaChatRequestBuilder {
        BetaChatRequestBuilder::default()
            .client(get_client())
            .model("deepseek-v4-flash")
            .max_tokens(32_u32)
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
