//! Protocol Buffers utilities for [`db`][`crate::db`] module.

use crate::error::Error;
use crate::protos::{Deserialize, Serialize};
use crate::protos::database::{
    AttributeValue as ProtosAttributeValue,
    attribute_value as protos_attribute_value,
};

use super::AttributeValue;

impl Serialize<ProtosAttributeValue> for AttributeValue {
    fn serialize(&self) -> Result<ProtosAttributeValue, Error> {
        let mut value = ProtosAttributeValue::new();
        match self {
            AttributeValue::String(s) => {
                value.value = Some(
                    protos_attribute_value::Value::StringValue(s.clone()),
                );
            }
        };
        Ok(value)
    }
}

impl Deserialize<AttributeValue> for ProtosAttributeValue {
    fn deserialize(self) -> Result<AttributeValue, Error> {
        if let Some(value) = self.value {
            match value {
                protos_attribute_value::Value::StringValue(s) => {
                    Ok(AttributeValue::String(s))
                },
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
            Some(protos_attribute_value::Value::StringValue("string".to_string())),
        );
    }

    #[test]
    fn attribute_value_string_can_be_deserialized_from_attribute_value_message() {
        let mut input = ProtosAttributeValue::new();
        input.value = Some(protos_attribute_value::Value::StringValue("string".to_string()));
        let output = input.deserialize().unwrap();
        assert_eq!(output, AttributeValue::String("string".to_string()));
    }
}
