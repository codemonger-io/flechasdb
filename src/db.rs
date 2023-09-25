//! Synchronous vector database.
//!
//! Use `build` submodule to build a new database.
//!
//! Use `stored` submodule to load a stored database.

use std::collections::HashMap;
use uuid::Uuid;

pub mod build;
pub mod proto;
pub mod stored;

/// Attributes associated with a vector.
pub type Attributes = HashMap<String, AttributeValue>;

/// Attribute table.
pub type AttributeTable = HashMap<Uuid, Attributes>;

/// Attribute value.
#[derive(Clone, Debug, PartialEq)]
pub enum AttributeValue {
    String(String),
}

impl From<String> for AttributeValue {
    fn from(s: String) -> Self {
        AttributeValue::String(s)
    }
}
