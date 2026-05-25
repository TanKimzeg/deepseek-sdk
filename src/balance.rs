//! User balance API.
//!
//! Maps to `GET /user/balance`.
use crate::DeepSeekClient;
use crate::DeepSeekError;
use crate::api_get;
use serde::Deserialize;

/// Account balance response.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize)]
pub struct Balance {
    pub is_available: bool,
    pub balance_infos: Vec<BalanceInfo>,
}

/// Balance entry by currency.
#[derive(Debug, Clone, Eq, PartialEq, Deserialize)]
pub struct BalanceInfo {
    pub currency: String,
    pub total_balance: String,
    pub granted_balance: String,
    pub topped_up_balance: String,
}

impl Balance {
    /// Fetch account balance.
    pub async fn get(client: DeepSeekClient) -> Result<Self, DeepSeekError> {
        api_get("/user/balance", client).await
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
    async fn test_get_balance() {
        let client = get_client();
        let balance = Balance::get(client).await.unwrap();
        println!("{:#?}", balance);
        assert!(balance.is_available);
        assert!(!balance.balance_infos.is_empty());
    }
}
