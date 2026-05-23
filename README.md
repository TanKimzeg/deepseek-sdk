## deepseek-rs

DeepSeek API client for Rust.

### Features

- Chat completions (`/chat/completions`)
- FIM completions (beta, `/beta/completions`)
- List models (`/models`)
- Account balance (`/user/balance`)
- Streaming via async receiver or blocking iterator

### Install

Add to `Cargo.toml`:

```toml
deepseek-rs = "0.1"
```

### API Key

Set your API key before running examples:

```bash
export DEEPSEEK_API="sk-..."
```

### Quick Start (Chat)

```rust
use deepseek_rs::chat::request::{ChatMessage, ChatRequestBuilder, Thinking};
use deepseek_rs::{Credentials, DeepSeekRequest, DEFAULT_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let req = ChatRequestBuilder::default()
  .credentials(Credentials::new("sk-...", DEFAULT_BASE_URL.clone()))
  .model("deepseek-v4-flash")
  .message(ChatMessage::User {
   content: "Hi".to_string(),
   name: None,
  })
  .thinking(Thinking::disabled())
  .max_tokens(1024)
  .build()?;

 let resp = req.send().await?;
 println!("{:#?}", resp);
 Ok(())
}
```

### Async Streaming

```rust
use deepseek_rs::chat::request::{ChatMessage, ChatRequestBuilder, Thinking};
use deepseek_rs::{Credentials, DeepSeekRequest, DEFAULT_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let req = ChatRequestBuilder::default()
  .credentials(Credentials::new("sk-...", DEFAULT_BASE_URL.clone()))
  .model("deepseek-v4-flash")
  .message(ChatMessage::User { content: "Hi".into(), name: None })
  .thinking(Thinking::disabled())
  .build()?;

 let mut rx = req.stream().await?;
 while let Some(item) = rx.recv().await {
  let chunk = item?;
  for choice in chunk.choices {
   if let Some(delta) = choice.delta.content {
    print!("{delta}");
   }
  }
 }
 Ok(())
}
```

### Blocking Streaming

```rust
use deepseek_rs::chat::request::{ChatMessage, ChatRequestBuilder, Thinking};
use deepseek_rs::{Credentials, DeepSeekRequest, DEFAULT_BASE_URL};

fn main() -> Result<(), Box<dyn std::error::Error>> {
 let req = ChatRequestBuilder::default()
  .credentials(Credentials::new("sk-...", DEFAULT_BASE_URL.clone()))
  .model("deepseek-v4-flash")
  .message(ChatMessage::User { content: "Hi".into(), name: None })
  .thinking(Thinking::disabled())
  .build()?;

 let mut stream = req.stream_blocking()?;
 for item in stream.by_ref() {
  let chunk = item?;
  for choice in chunk.choices {
   if let Some(delta) = choice.delta.content {
    print!("{delta}");
   }
  }
 }
 Ok(())
}
```

### FIM Completion (Beta)

FIM uses the beta base URL.

```rust
use deepseek_rs::completion::fim::FIMCompletionRequestBuilder;
use deepseek_rs::{Credentials, DeepSeekRequest, DEFAULT_BETA_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let req = FIMCompletionRequestBuilder::default()
  .credentials(Credentials::new("sk-...", DEFAULT_BETA_BASE_URL.clone()))
  .model("deepseek-v4-pro")
  .prompt("def fib(n):")
  .suffix("    return fib(n-1) + fib(n-2)")
  .max_tokens(128)
  .build()?;

 let resp = req.send().await?;
 println!("{:#?}", resp);
 Ok(())
}
```

### List Models

```rust
use deepseek_rs::models::Models;
use deepseek_rs::{Credentials, DEFAULT_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let credentials = Credentials::new("sk-...", DEFAULT_BASE_URL.clone());
 let models = Models::list(credentials).await?;
 println!("{:#?}", models);
 Ok(())
}
```

### Balance

```rust
use deepseek_sdk::balance::Balance;
use deepseek_sdk::{Credentials, DEFAULT_BASE_URL};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
 let credentials = Credentials::new("sk-...", DEFAULT_BASE_URL.clone());
 let balance = Balance::get(credentials).await?;
 println!("{:#?}", balance);
 Ok(())
}
```

### Error Handling

All requests return `DeepSeekError` on failure, covering:

- API error payloads (`Api`)
- HTTP errors (`Http`)
- Decode errors (`Decode`)
- Transport failures (`Transport`)

### License

MIT
