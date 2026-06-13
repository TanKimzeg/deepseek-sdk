//! DeepSeek API client for Rust.
//!
//! This crate provides:
//! - Chat completions (`/chat/completions`)
//! - FIM completions (beta, `/beta/completions`)
//! - Model listing (`/models`)
//! - Account balance (`/user/balance`)
//!
//! Streaming is supported in both async and blocking forms. The async API returns
//! a `tokio::mpsc::Receiver`, while the blocking API returns an iterator that
//! yields stream items.
//!
//! ```no_run
//! use deepseek_sdk::chat::request::{ChatMessage, ChatRequestBuilder, Thinking};
//! use deepseek_sdk::{DeepSeekClient, DeepSeekRequest, DEFAULT_BASE_URL};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let req = ChatRequestBuilder::default()
//!     .client(DeepSeekClient::new("sk-...", DEFAULT_BASE_URL.clone()))
//!     .model("deepseek-v4-flash")
//!     .message(ChatMessage::User { content: "Hi".into(), name: None })
//!     .thinking(Thinking::disabled())
//!     .build()?;
//! let _resp = req.send().await?;
//! # Ok(()) }
//! ```
pub mod balance;
pub mod chat;
pub mod completion;
pub mod error;
pub mod models;

use crate::error::{ApiErrorEnvelope, DeepSeekError};

use reqwest::{Client, Method, RequestBuilder, Response, header::AUTHORIZATION};
use reqwest_eventsource::{EventSource, RequestBuilderExt};
use serde::{Serialize, de::DeserializeOwned};
use std::future::Future;
use std::sync::LazyLock;
use tokio::sync::mpsc;

/// Default base URL for stable API endpoints.
pub static DEFAULT_BASE_URL: LazyLock<String> =
    LazyLock::new(|| String::from("https://api.deepseek.com"));
/// Default base URL for beta endpoints (e.g. FIM completion).
pub static DEFAULT_BETA_BASE_URL: LazyLock<String> =
    LazyLock::new(|| String::from("https://api.deepseek.com/beta"));

/// API credentials for a DeepSeek endpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
struct Credentials {
    pub(crate) api_key: String,
    pub(crate) base_url: String,
}

#[derive(Clone, Debug)]
pub struct DeepSeekClient {
    pub(crate) credentials: Credentials,
    pub client: Client,
}

impl PartialEq for DeepSeekClient {
    fn eq(&self, other: &Self) -> bool {
        self.credentials == other.credentials
    }
}

impl Eq for DeepSeekClient {}

impl DeepSeekClient {
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        DeepSeekClient {
            credentials: Credentials::new(api_key, base_url),
            client: Client::new(),
        }
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    pub fn with_credentials(
        mut self,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        self.credentials = Credentials::new(api_key, base_url);
        self
    }
}

impl Credentials {
    /// Create credentials with an API key and base URL.
    pub(crate) fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Credentials {
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }
}

/// Unified request interface for DeepSeek endpoints.
///
/// Requests that support streaming should return stream items through `stream`
/// and a blocking iterator through `stream_blocking`.
pub trait DeepSeekRequest: Sized {
    /// Full response type for non-streaming calls.
    type Response;

    /// Item type emitted by streaming calls.
    type StreamItem;
    /// Blocking stream iterator type.
    type BlockingStream: Iterator<Item = Self::StreamItem>;

    /// Send a non-streaming request.
    fn send(self) -> impl Future<Output = Result<Self::Response, DeepSeekError>> + Send;
    /// Send a streaming request (SSE), returning a receiver of stream items.
    fn stream(
        self,
    ) -> impl Future<Output = Result<mpsc::Receiver<Self::StreamItem>, DeepSeekError>> + Send;
    /// Send a streaming request but consume results via a blocking iterator.
    fn stream_blocking(self) -> Result<Self::BlockingStream, DeepSeekError>;
}

async fn api_request_json<F, T>(
    method: Method,
    route: &str,
    builder: F,
    deepseek_client: DeepSeekClient,
) -> Result<T, DeepSeekError>
where
    F: FnOnce(RequestBuilder) -> RequestBuilder,
    T: DeserializeOwned,
{
    let response = api_request(method, route, builder, deepseek_client).await?;
    let status = response.status();

    let text = response.text().await?;

    if !status.is_success() {
        if let Ok(envelope) = serde_json::from_str::<ApiErrorEnvelope>(&text) {
            return Err(DeepSeekError::api(
                envelope.error,
                Some(status.as_u16()),
                Some(text),
            ));
        }

        return Err(DeepSeekError::http(status.as_u16(), text));
    }

    serde_json::from_str::<T>(&text).map_err(|err| DeepSeekError::decode(err.to_string(), text))
}

async fn api_request<F>(
    method: Method,
    route: &str,
    builder: F,
    deepseek_client: DeepSeekClient,
) -> Result<Response, DeepSeekError>
where
    F: FnOnce(RequestBuilder) -> RequestBuilder,
{
    let client = deepseek_client.client;
    let mut request = client.request(
        method,
        format!("{}{route}", deepseek_client.credentials.base_url),
    );
    request = builder(request);
    let response = request
        .header(
            AUTHORIZATION,
            format!("Bearer {}", deepseek_client.credentials.api_key),
        )
        .send()
        .await?;
    Ok(response)
}

async fn api_request_stream<F>(
    method: Method,
    route: &str,
    builder: F,
    deepseek_client: DeepSeekClient,
) -> Result<EventSource, DeepSeekError>
where
    F: FnOnce(RequestBuilder) -> RequestBuilder,
{
    let mut request = deepseek_client.client.request(
        method,
        format!("{}{route}", deepseek_client.credentials.base_url),
    );
    request = builder(request);
    let stream = request
        .header(
            AUTHORIZATION,
            format!("Bearer {}", deepseek_client.credentials.api_key),
        )
        .eventsource()
        .map_err(|err| DeepSeekError::decode(err.to_string(), String::new()))?;
    Ok(stream)
}

async fn api_get<T>(route: &str, client: DeepSeekClient) -> Result<T, DeepSeekError>
where
    T: DeserializeOwned,
{
    api_request_json(Method::GET, route, |request| request, client).await
}

async fn api_post<J, T>(route: &str, json: &J, client: DeepSeekClient) -> Result<T, DeepSeekError>
where
    J: Serialize + ?Sized,
    T: DeserializeOwned,
{
    api_request_json(Method::POST, route, |request| request.json(json), client).await
}
