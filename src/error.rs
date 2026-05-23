//! Error types for DeepSeek API interactions.
use serde::Deserialize;
use std::error::Error;
use std::fmt;

/// Error payload returned by DeepSeek APIs.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize)]
pub struct ApiError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    /// Present only for API error payloads.
    pub param: Option<String>,
    /// Present only for API error payloads.
    pub code: Option<String>,
}

/// Envelope used by some API endpoints: `{ "error": { ... } }`.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize)]
pub(crate) struct ApiErrorEnvelope {
    pub error: ApiError,
}

/// Categorized reqwest error kinds for diagnostics.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ReqwestErrorKind {
    Decode,
    Timeout,
    Connect,
    Request,
    Body,
    Status,
}

impl ReqwestErrorKind {
    fn as_str(&self) -> &'static str {
        match self {
            ReqwestErrorKind::Decode => "decode",
            ReqwestErrorKind::Timeout => "timeout",
            ReqwestErrorKind::Connect => "connect",
            ReqwestErrorKind::Request => "request",
            ReqwestErrorKind::Body => "body",
            ReqwestErrorKind::Status => "status",
        }
    }
}

/// Transport-level failure from reqwest.
#[derive(Debug)]
pub struct TransportError {
    pub source: reqwest::Error,
    pub kind: Option<ReqwestErrorKind>,
}

/// Unified error type for this crate.
#[derive(Debug)]
pub enum DeepSeekError {
    /// API returned a structured error payload.
    Api {
        error: ApiError,
        status: Option<u16>,
        body: Option<String>,
    },
    /// Non-JSON or otherwise unrecognized HTTP error.
    Http {
        status: u16,
        body: Option<String>,
    },
    /// Response could not be decoded into the expected schema.
    Decode {
        message: String,
        body: Option<String>,
    },
    /// Transport errors from reqwest.
    Transport(TransportError),
}

impl DeepSeekError {
    pub(crate) fn api(error: ApiError, status: Option<u16>, body: Option<String>) -> Self {
        DeepSeekError::Api {
            error,
            status,
            body,
        }
    }

    pub(crate) fn http(status: u16, body: String) -> Self {
        DeepSeekError::Http {
            status,
            body: Some(body),
        }
    }

    pub(crate) fn decode(message: String, body: String) -> Self {
        DeepSeekError::Decode {
            message,
            body: Some(body),
        }
    }
}

impl fmt::Display for DeepSeekError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeepSeekError::Api { error, status, .. } => write!(
                f,
                "DeepSeek API error: {} (type={}, param={:?}, code={:?}, status={:?})",
                error.message, error.error_type, error.param, error.code, status
            ),
            DeepSeekError::Http { status, body } => {
                write!(f, "HTTP error: status={}, body={:?}", status, body)
            }
            DeepSeekError::Decode { message, body } => {
                write!(f, "Decode error: {} (body={:?})", message, body)
            }
            DeepSeekError::Transport(transport) => {
                if let Some(kind) = &transport.kind {
                    write!(f, "reqwest {} error: {}", kind.as_str(), transport.source)
                } else {
                    write!(f, "reqwest error: {}", transport.source)
                }
            }
        }
    }
}

impl Error for DeepSeekError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            DeepSeekError::Transport(transport) => Some(&transport.source),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for DeepSeekError {
    fn from(value: reqwest::Error) -> Self {
        let kind = if value.is_decode() {
            Some(ReqwestErrorKind::Decode)
        } else if value.is_timeout() {
            Some(ReqwestErrorKind::Timeout)
        } else if value.is_connect() {
            Some(ReqwestErrorKind::Connect)
        } else if value.is_request() {
            Some(ReqwestErrorKind::Request)
        } else if value.is_body() {
            Some(ReqwestErrorKind::Body)
        } else if value.is_status() {
            Some(ReqwestErrorKind::Status)
        } else {
            None
        };

        DeepSeekError::Transport(TransportError {
            source: value,
            kind,
        })
    }
}
