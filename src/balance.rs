//! User balance API.
//!
//! Maps to `GET /user/balance`.
use crate::Credentials;
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
    pub async fn get(credentials: Credentials) -> Result<Self, DeepSeekError> {
        api_get("/user/balance", Some(credentials)).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_BASE_URL;
    fn get_credentials() -> Credentials {
        Credentials::new(
            std::env::var("DEEPSEEK_API").expect("DEEPSEEK_API is not set"),
            DEFAULT_BASE_URL.clone(),
        )
    }

    #[tokio::test]
    async fn test_get_balance() {
        let credentials = get_credentials();
        let balance = Balance::get(credentials).await.unwrap();
        println!("{:#?}", balance);
        assert!(balance.is_available);
        assert!(!balance.balance_infos.is_empty());
    }
}
