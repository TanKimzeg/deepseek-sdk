//! Model list API.
//!
//! Maps to `GET /models`.
use crate::Credentials;
use crate::DeepSeekError;
use crate::api_get;
use serde::Deserialize;

/// Model list response.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize)]
pub struct Models {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model info entry.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub owned_by: String,
}

impl Models {
    /// List available models.
    pub async fn list(credentials: Credentials) -> Result<Self, DeepSeekError> {
        api_get("/models", Some(credentials)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Credentials;
    use crate::DEFAULT_BASE_URL;

    fn get_credentials() -> Credentials {
        Credentials::new(
            std::env::var("DEEPSEEK_API").expect("DEEPSEEK_API is not set"),
            DEFAULT_BASE_URL.clone(),
        )
    }

    #[tokio::test]
    async fn test_list_models() {
        let credentials = get_credentials();
        let models = Models::list(credentials).await.unwrap();
        println!("{:#?}", models);
        assert_eq!(models.object, "list");
        assert!(!models.data.is_empty());
    }
}
