//! Chat completion request/response models.
//!
//! This module contains the data structures for the `/chat/completions` API
//! and re-exports streaming helpers from the client implementation.
use crate::DeepSeekClient;
use serde::{Deserialize, Serialize};

pub mod client;
pub use client::{ChatStreamBlocking, ChatStreamItem};

/// Helper to skip serialization of empty `Vec` fields wrapped in `Option`.
pub(crate) fn is_none_or_empty_vec<T>(opt: &Option<Vec<T>>) -> bool {
    opt.as_ref().map(|v| v.is_empty()).unwrap_or(true)
}

/// Non-streaming chat completion response type.
pub type Chat = response::ChatGeneric<response::ChatChoice>;

/// Streaming chat completion response type (SSE chunks).
pub type ChatStream = response::ChatGeneric<response::ChatChoiceStream>;

pub mod response {
    use super::*;
    /// Token usage statistics for a request.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct Usage {
        pub completion_tokens: u64,
        pub prompt_tokens: u64,
        pub prompt_cache_hit_tokens: u64,
        pub prompt_cache_miss_tokens: u64,
        pub total_tokens: u64,
        pub completion_tokens_details: Option<CompletionTokensDetails>,
    }
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct CompletionTokensDetails {
        pub reasoning_tokens: u64,
    }

    /// Generic chat response container.
    #[derive(Clone, Debug, PartialEq, Deserialize)]
    pub struct ChatGeneric<C> {
        pub id: String,
        pub choices: Vec<C>,
        pub created: u64,
        pub model: String,
        pub system_fingerprint: String,
        pub object: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage: Option<Usage>,
    }

    /// Non-streaming choice result.
    #[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
    pub struct ChatChoice {
        pub finish_reason: FinishReason,
        pub index: u64,
        pub message: ChoiceMessage,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<Logprobs>,
    }

    /// Streaming choice delta.
    #[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
    pub struct ChatChoiceStream {
        pub finish_reason: Option<FinishReason>,
        pub index: u64,
        pub delta: ChoiceMessageDelta,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub logprobs: Option<Logprobs>,
    }

    /// Assistant message content in non-streaming responses.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct ChoiceMessage {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "is_none_or_empty_vec")]
        pub tool_calls: Option<Vec<ToolCall>>,
        pub role: Role,
    }

    /// Assistant message delta in streaming responses.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct ChoiceMessageDelta {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reasoning_content: Option<String>,
        #[serde(skip_serializing_if = "is_none_or_empty_vec")]
        pub tool_calls: Option<Vec<ToolCall>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub role: Option<Role>,
    }

    /// Role of a chat message.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Role {
        System,
        User,
        Assistant,
        Tool,
    }

    /// Tool call emitted by the model.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct ToolCall {
        pub id: String,
        #[serde(rename = "type")]
        pub typ: ToolCallType,
        pub function: ToolCallFunction,
    }

    impl ToolCall {
        /// Build a function tool call with an id, name, and arguments JSON string.
        pub fn new(
            id: impl Into<String>,
            name: impl Into<String>,
            arguments: impl Into<String>,
        ) -> Self {
            ToolCall {
                id: id.into(),
                typ: ToolCallType::Function,
                function: ToolCallFunction {
                    name: name.into(),
                    arguments: arguments.into(),
                },
            }
        }
    }

    /// Tool call type.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ToolCallType {
        Function,
    }

    /// Tool call function payload.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct ToolCallFunction {
        pub name: String,
        pub arguments: String,
    }
    /// Reason for completion termination.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum FinishReason {
        Stop,
        Length,
        ContentFilter,
        ToolCalls,
        InsufficientSystemResources,
    }
    /// Token-level log probability data.
    #[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
    pub struct Logprobs {
        #[serde(skip_serializing_if = "is_none_or_empty_vec")]
        pub content: Option<Vec<LogprobsContent>>,
        #[serde(skip_serializing_if = "is_none_or_empty_vec")]
        pub reasoning_content: Option<Vec<LogprobsReasoningContent>>,
    }
    /// Logprobs for content tokens.
    #[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
    pub struct LogprobsContent {
        pub token: String,
        pub logprob: f64,
        pub bytes: Option<Vec<u8>>,
        pub top_logprobs: Vec<TopLogprobs>,
    }

    /// Top logprob candidates for a token.
    #[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
    pub struct TopLogprobs {
        pub token: String,
        pub logprob: f64,
        pub bytes: Option<Vec<u8>>,
    }
    /// Logprobs for reasoning tokens.
    #[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
    pub struct LogprobsReasoningContent {
        pub token: String,
        pub logprob: f64,
        pub bytes: Option<Vec<u8>>,
        pub top_logprobs: Vec<TopLogprobs>,
    }
}

/// Request payloads for `/chat/completions`.
pub mod request {
    use super::*;
    use derive_builder::Builder;
    pub(crate) fn is_none_or_empty_stop(opt: &Option<Stop>) -> bool {
        opt.as_ref().map(|stop| stop.is_empty()).unwrap_or(true)
    }

    /// Chat completion request body.
    #[derive(Clone, Debug, Serialize, Builder)]
    #[builder(
        pattern = "owned",
        setter(into, strip_option),
        build_fn(validate = "Self::validate"),
        name = "ChatRequestBuilder"
    )]
    pub struct ChatRequest {
        #[serde(skip_serializing)]
        pub client: DeepSeekClient,

        #[builder(setter(each(name = "message", into)))]
        pub messages: Vec<ChatMessage>,
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
    /// Chat message variants.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(tag = "role", rename_all = "snake_case")]
    pub enum ChatMessage {
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

            #[serde(skip_serializing_if = "super::is_none_or_empty_vec")]
            tool_calls: Option<Vec<super::response::ToolCall>>,
        },
        Tool {
            content: String,
            tool_call_id: String,
        },
    }
    /// Reasoning effort hints for the model.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ReasoningEffort {
        High,
        Max,
    }
    /// Response format configuration.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct ResponseFormat {
        /// text 或 json_object。
        #[serde(rename = "type")]
        pub(crate) typ: ResponseFormatType,
    }
    /// Supported response format types.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub(crate) enum ResponseFormatType {
        Text,
        JsonObject,
    }

    impl ResponseFormat {
        pub fn text() -> Self {
            ResponseFormat {
                typ: ResponseFormatType::Text,
            }
        }

        pub fn json_object() -> Self {
            ResponseFormat {
                typ: ResponseFormatType::JsonObject,
            }
        }
    }

    /// Stop sequences for generation.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(untagged)]
    pub enum Stop {
        One(String),
        Many(Vec<String>),
    }

    impl Stop {
        fn is_empty(&self) -> bool {
            match self {
                Stop::One(value) => value.is_empty(),
                Stop::Many(values) => values.is_empty(),
            }
        }
    }

    impl From<String> for Stop {
        fn from(value: String) -> Self {
            Stop::One(value)
        }
    }

    impl From<&str> for Stop {
        fn from(value: &str) -> Self {
            Stop::One(value.to_string())
        }
    }

    impl<T> From<Vec<T>> for Stop
    where
        T: Into<String>,
    {
        fn from(values: Vec<T>) -> Self {
            Stop::Many(values.into_iter().map(Into::into).collect())
        }
    }
    /// Streaming options for SSE responses.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct StreamOptions {
        pub include_usage: bool,
    }
    /// Tool definition used by the model.
    #[derive(Clone, Debug, PartialEq, Eq, Serialize)]
    pub struct Tool {
        #[serde(rename = "type")]
        pub typ: ToolType,
        pub function: ToolFunctionDefinition,
    }

    impl Tool {
        pub fn new(
            name: impl Into<String>,
            description: impl Into<String>,
            parameters: Option<serde_json::Value>,
        ) -> Self {
            Tool {
                typ: ToolType::Function,
                function: ToolFunctionDefinition {
                    name: name.into(),
                    description: description.into(),
                    parameters,
                },
            }
        }
    }

    /// Tool type.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ToolType {
        Function,
    }

    /// Tool function definition.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct ToolFunctionDefinition {
        pub description: String,
        pub name: String,
        pub parameters: Option<serde_json::Value>,
    }
    /// Tool choice configuration.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(untagged)]
    pub enum ToolChoice {
        /// "none" | "auto" | "required"
        Simple(ChatToolChoice),
        /// {"type":"function","function":{...}}
        Named(ChatNamedToolChoice),
    }

    impl ToolChoice {
        pub fn named(function: serde_json::Value) -> Self {
            ToolChoice::Named(ChatNamedToolChoice {
                typ: ToolType::Function,
                function,
            })
        }

        pub fn none() -> Self {
            ToolChoice::Simple(ChatToolChoice::None)
        }

        pub fn auto() -> Self {
            ToolChoice::Simple(ChatToolChoice::Auto)
        }

        pub fn required() -> Self {
            ToolChoice::Simple(ChatToolChoice::Required)
        }
    }

    /// Tool choice values.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ChatToolChoice {
        None,
        Auto,
        Required,
    }
    /// Named tool choice configuration.
    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct ChatNamedToolChoice {
        #[serde(rename = "type")]
        pub typ: ToolType,
        pub function: serde_json::Value,
    }

    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    pub struct Thinking {
        /// enabled / disabled
        #[serde(rename = "type")]
        pub(crate) typ: ThinkingType,
    }

    impl Thinking {
        pub fn enabled() -> Self {
            Thinking {
                typ: ThinkingType::Enabled,
            }
        }

        pub fn disabled() -> Self {
            Thinking {
                typ: ThinkingType::Disabled,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
    #[serde(rename_all = "snake_case")]
    pub(crate) enum ThinkingType {
        Enabled,
        Disabled,
    }

    impl ChatRequestBuilder {
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

            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::request::*;
    use super::response::*;
    use serde_json::{Value, json};

    #[test]
    fn response_format_serializes_to_json_object() {
        let format = ResponseFormat::json_object();
        let value = serde_json::to_value(format).unwrap();
        assert_eq!(value, json!({"type": "json_object"}));
    }

    #[test]
    fn stop_supports_string_and_array() {
        let single = Stop::from("END");
        let many = Stop::from(vec!["END", "STOP"]);

        let single_value = serde_json::to_value(single).unwrap();
        let many_value = serde_json::to_value(many).unwrap();

        assert_eq!(single_value, json!("END"));
        assert_eq!(many_value, json!(["END", "STOP"]));

        let single_back: Stop = serde_json::from_value(json!("END")).unwrap();
        let many_back: Stop = serde_json::from_value(json!(["A", "B"])).unwrap();
        assert!(matches!(single_back, Stop::One(_)));
        assert!(matches!(many_back, Stop::Many(_)));

        let none_back: Option<Stop> = serde_json::from_value(Value::Null).unwrap();
        assert!(none_back.is_none());
    }

    #[test]
    fn tool_choice_serializes_simple_and_named() {
        let simple = ToolChoice::Simple(ChatToolChoice::Auto);
        let simple_value = serde_json::to_value(simple).unwrap();
        assert_eq!(simple_value, json!("auto"));

        let named = ToolChoice::named(json!({"name": "get_weather"}));
        let named_value = serde_json::to_value(named).unwrap();
        assert_eq!(
            named_value,
            json!({"type": "function", "function": {"name": "get_weather"}})
        );
    }

    #[test]
    fn chat_message_serializes_role_and_omits_prefix_by_default() {
        let message = ChatMessage::Assistant {
            content: Some("Hello".to_string()),
            name: None,
            tool_calls: None,
        };
        let value = serde_json::to_value(message).unwrap();
        assert_eq!(value.get("role"), Some(&json!("assistant")));
        assert_eq!(value.get("content"), Some(&json!("Hello")));
        assert!(value.get("reasoning_content").is_none());
    }

    #[test]
    fn response_tool_call_type_serializes_as_function() {
        let call = ToolCall::new("call_i", "get_weather", "{}");
        let value = serde_json::to_value(call).unwrap();
        assert_eq!(value.get("type"), Some(&json!("function")));
    }

    #[test]
    fn builder_validation_rejects_out_of_range_values() {
        fn base_builder() -> ChatRequestBuilder {
            ChatRequestBuilder::default()
                .model("deepseek-v4-pro")
                .message(ChatMessage::User {
                    content: "Hi".to_string(),
                    name: None,
                })
        }

        let too_hot = base_builder().temperature(2.5).build();
        assert!(too_hot.is_err());

        let bad_top_p = base_builder().top_p(1.1).build();
        assert!(bad_top_p.is_err());

        let bad_top_logprobs = base_builder()
            .top_logprobs(21 as u32)
            .logprobs(true)
            .build();
        assert!(bad_top_logprobs.is_err());

        let missing_logprobs = base_builder().top_logprobs(2 as u32).build();
        assert!(missing_logprobs.is_err());
    }

    #[test]
    fn thinking_struct_serializes_type() {
        let thinking = Thinking::disabled();
        let value = serde_json::to_value(&thinking).unwrap();
        assert_eq!(value.get("type"), Some(&json!("disabled")));

        let req = ChatRequestBuilder::default()
            .model("deepseek-v4-flash")
            .message(ChatMessage::User {
                content: "Hi".to_string(),
                name: None,
            })
            .thinking(thinking)
            .reasoning_effort(ReasoningEffort::Max)
            .build();
        // thinking options type cannot be disabled when reasoning_effort is set
        assert!(req.is_err());
    }
}
