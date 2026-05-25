//! Model list API.
//!
//! Maps to `GET /models`.
use crate::DeepSeekClient;
use crate::DeepSeekError;
use crate::api_get;
use serde::Deserialize;

/// Model list response.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize)]
pub struct Models {
    /// Possible values: [`list`]
    pub object: String,
    pub data: Vec<ModelInfo>,
}

/// Model info entry.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize)]
pub struct ModelInfo {
    /// The model identifier, which can be referenced in the API endpoints.
    pub id: String,

    /// Possible values: [model]
    ///
    /// The object type, which is always "model".
    pub object: String,

    /// The organization that owns the model.
    pub owned_by: String,
}

impl Models {
    /// List available models.
    pub async fn list(client: DeepSeekClient) -> Result<Self, DeepSeekError> {
        api_get("/models", client).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_BASE_URL;

    fn get_client() -> DeepSeekClient {
        DeepSeekClient::new(
            std::env::var("DEEPSEEK_API").expect("DEEPSEEK_API is not set"),
            DEFAULT_BASE_URL.clone(),
        )
    }

    #[tokio::test]
    async fn test_list_models() {
        let client = get_client();
        let models = Models::list(client).await.unwrap();
        println!("{:#?}", models);
        assert_eq!(models.object, "list");
        assert!(!models.data.is_empty());
    }
}
