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
    /// String value.
    String(String),
    /// 64-bit unsigned integer value.
    Uint64(u64),
}

impl From<String> for AttributeValue {
    fn from(s: String) -> Self {
        AttributeValue::String(s)
    }
}

impl From<&str> for AttributeValue {
    fn from(s: &str) -> Self {
        AttributeValue::String(s.to_string())
    }
}

impl From<u64> for AttributeValue {
    fn from(n: u64) -> Self {
        AttributeValue::Uint64(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_value_can_be_made_from_string() {
        assert_eq!(
            AttributeValue::String("attr".to_string()),
            "attr".to_string().into(),
        );
        assert_eq!(
            AttributeValue::String("".to_string()),
            "".to_string().into(),
        );
    }

    #[test]
    fn attribute_value_can_be_made_from_str_ref() {
        assert_eq!(AttributeValue::String("attr".to_string()), "attr".into());
        assert_eq!(AttributeValue::String("".to_string()), "".into());
    }

    #[test]
    fn attribute_value_can_be_made_from_u64() {
        assert_eq!(AttributeValue::Uint64(0), 0u64.into());
        assert_eq!(
            AttributeValue::Uint64(0xFFFF_FFFF_FFFF_FFFF),
            0xFFFF_FFFF_FFFF_FFFFu64.into(),
        );
    }
}
