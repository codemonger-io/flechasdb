//! Protocol Buffers utilities for `vector`.

use crate::error::Error;
use crate::protos::{Deserialize, Serialize};
use crate::protos::database::{
    EncodedVectorSet as ProtosEncodedVectorSet,
    VectorSet as ProtosVectorSet,
};

use super::BlockVectorSet;

impl Serialize<ProtosVectorSet> for BlockVectorSet<f32> {
    fn serialize(&self) -> Result<ProtosVectorSet, Error> {
        let mut vs = ProtosVectorSet::new();
        vs.vector_size = self.vector_size() as u32;
        vs.data = self.data.clone();
        Ok(vs)
    }
}

impl Deserialize<BlockVectorSet<f32>> for ProtosVectorSet {
    fn deserialize(self) -> Result<BlockVectorSet<f32>, Error> {
        BlockVectorSet::chunk(
            self.data,
            (self.vector_size as usize)
                .try_into()
                .or(Err(Error::InvalidData(
                    "vector size must not be zero".to_string(),
                )))?,
        )
    }
}

impl Serialize<ProtosEncodedVectorSet> for BlockVectorSet<u32> {
    fn serialize(&self) -> Result<ProtosEncodedVectorSet, Error> {
        let mut vs = ProtosEncodedVectorSet::new();
        vs.vector_size = self.vector_size() as u32;
        vs.data = self.data.clone();
        Ok(vs)
    }
}

impl Deserialize<BlockVectorSet<u32>> for ProtosEncodedVectorSet {
    fn deserialize(self) -> Result<BlockVectorSet<u32>, Error> {
        BlockVectorSet::chunk(
            self.data,
            (self.vector_size as usize)
                .try_into()
                .or(Err(Error::InvalidData(
                    "vector size must not be zero".to_string(),
                )))?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_vector_set_f32_can_be_serialized_as_vector_set_message() {
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input: BlockVectorSet<f32> = BlockVectorSet::chunk(
            data.clone(),
            2.try_into().unwrap(),
        ).unwrap();
        let output = input.serialize().unwrap();
        assert_eq!(output.vector_size, 2);
        assert_eq!(output.data, data);
    }

    #[test]
    fn block_vector_set_f32_can_be_deserialized_from_vector_set_message() {
        let mut input = ProtosVectorSet::new();
        input.vector_size = 2;
        input.data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let output = input.deserialize().unwrap();
        assert_eq!(output.vector_size(), 2);
        assert_eq!(output.len(), 3);
        assert_eq!(output.get(0), vec![0.0, 1.0]);
        assert_eq!(output.get(1), vec![2.0, 3.0]);
        assert_eq!(output.get(2), vec![4.0, 5.0]);
    }

    #[test]
    fn block_vector_set_f32_cannot_be_deserialized_if_vector_size_is_zero() {
        let mut input = ProtosVectorSet::new();
        input.vector_size = 0;
        assert!(input.deserialize().is_err());
    }

    #[test]
    fn block_vector_set_u32_can_be_serialized_as_encoded_vector_set_message() {
        let data: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
        let input: BlockVectorSet<u32> = BlockVectorSet::chunk(
            data.clone(),
            3.try_into().unwrap(),
        ).unwrap();
        let output = input.serialize().unwrap();
        assert_eq!(output.vector_size, 3);
        assert_eq!(output.data, data);
    }

    #[test]
    fn block_vector_set_u32_can_be_deserialized_from_encoded_vector_set_message() {
        let mut input = ProtosEncodedVectorSet::new();
        input.vector_size = 3;
        input.data = vec![1, 2, 3, 4, 5, 6];
        let output = input.deserialize().unwrap();
        assert_eq!(output.vector_size(), 3);
        assert_eq!(output.len(), 2);
        assert_eq!(output.get(0), vec![1, 2, 3]);
        assert_eq!(output.get(1), vec![4, 5, 6]);
    }

    #[test]
    fn block_vector_set_u32_cannot_be_deserialized_if_vector_size_is_zero() {
        let mut input = ProtosEncodedVectorSet::new();
        input.vector_size = 0;
        assert!(input.deserialize().is_err());
    }
}
