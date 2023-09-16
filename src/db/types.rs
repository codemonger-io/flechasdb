//! Types used in the database.

use std::collections::HashMap;

/// Attributes associated with a vector.
pub type Attributes = HashMap<String, AttributeValue>;

/// Attribute value.
#[derive(Clone, Debug)]
pub enum AttributeValue {
    String(String),
}

impl From<String> for AttributeValue {
    fn from(s: String) -> Self {
        AttributeValue::String(s)
    }
}
