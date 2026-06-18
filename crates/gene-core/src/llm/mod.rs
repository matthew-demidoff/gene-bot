//! LLM wire types and the incremental stream parser. The HTTP/SSE transport and
//! model discovery live in [`crate::provider`].

pub mod stream;
pub mod types;

pub use stream::{StreamEvent, StreamParser};
pub use types::WireMessage;
