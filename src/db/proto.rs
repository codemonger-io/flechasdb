//! Protocol Buffers utilities for [`db`][`crate::db`] module.

use crate::error::Error;
use crate::protos::{Deserialize, Serialize};
use crate::protos::database::{
    AttributeValue as ProtosAttributeValue,
    attribute_value::Value::{
        StringValue as ProtosStringValue,
        Uint64Value as ProtosUint64Value,
    },
};

use super::AttributeValue;

impl Serialize<ProtosAttributeValue> for AttributeValue {
    fn serialize(&self) -> Result<ProtosAttributeValue, Error> {
        let mut value = ProtosAttributeValue::new();
        value.value = match self {
            AttributeValue::String(s) => Some(ProtosStringValue(s.clone())),
            AttributeValue::Uint64(n) => Some(ProtosUint64Value(*n)),
        };
        Ok(value)
    }
}

impl Deserialize<AttributeValue> for ProtosAttributeValue {
    fn deserialize(self) -> Result<AttributeValue, Error> {
        if let Some(value) = self.value {
            match value {
                ProtosStringValue(s) => Ok(AttributeValue::String(s)),
                ProtosUint64Value(n) => Ok(AttributeValue::Uint64(n)),
            }
        } else {
            Err(Error::InvalidData(format!("missing attribute value")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attribute_value_string_can_be_serialized_as_attribute_value_message() {
        let input = AttributeValue::String("string".to_string());
        let output = input.serialize().unwrap();
        assert_eq!(
            output.value,
            Some(ProtosStringValue("string".to_string())),
        );
    }

    #[test]
    fn attribute_value_string_can_be_deserialized_from_attribute_value_message() {
        let mut input = ProtosAttributeValue::new();
        input.value = Some(ProtosStringValue("string".to_string()));
        let output = input.deserialize().unwrap();
        assert_eq!(output, AttributeValue::String("string".to_string()));
    }

    #[test]
    fn attribute_value_uint64_can_be_serialized_as_attribute_value_message() {
        let input = AttributeValue::Uint64(0x1234_5678_9ABC_DEF0u64);
        let output = input.serialize().unwrap();
        assert_eq!(
            output.value,
            Some(ProtosUint64Value(0x1234_5678_9ABC_DEF0u64)),
        );
    }

    #[test]
    fn attribute_value_uint64_can_be_deserialized_from_attribute_value_message() {
        let mut input = ProtosAttributeValue::new();
        input.value = Some(ProtosUint64Value(0x1234_5678_9ABC_DEF0u64));
        let output = input.deserialize().unwrap();
        assert_eq!(output, AttributeValue::Uint64(0x1234_5678_9ABC_DEF0u64));
    }

    #[test]
    fn attribute_value_message_without_value_cannot_be_deserialized() {
        assert!(ProtosAttributeValue::new().deserialize().is_err());
    }
}
