/// Serializes `Database` into a `protobuf::Message`.

use core::iter::IntoIterator;
use std::collections::HashMap;
use uuid::Uuid;

use crate::db::types::{AttributeValue, Attributes};
use crate::error::Error;
use crate::io::{FileSystem, HashedFileOut};
use crate::kmeans::Codebook;
use crate::protos::database::{
    AttributeValue as ProtosAttributeValue,
    AttributesLog as ProtosAttributesLog,
    Vector as ProtosVector,
    Codebook as ProtosCodebook,
    Database as ProtosDatabase,
    EncodedVector as ProtosEncodedVector,
    OperationSetAttribute as ProtosOperationSetAttribute,
    Partition as ProtosPartition,
    VectorSet as ProtosVectorSet,
    attribute_value as protos_attribute_value,
};
use crate::partitions::Partitions;
use crate::protos::{Deserialize, Serialize, write_message};
use crate::vector::{BlockVectorSet, VectorSet};
use super::{Database, Partition};

/// Extension of a Protocol Buffers file.
pub const PROTOBUF_EXTENSION: &str = "binpb";

/// Serializes `Database`.
pub fn serialize_database<'a, T, VS, FS>(
    db: &'a Database<T, VS>,
    fs: &mut FS,
) -> Result<(), Error>
where
    T: Clone,
    VS: VectorSet<T>,
    DatabaseSerialize<'a, T, VS>: Serialize<ProtosDatabase>,
    Partition<T>: Serialize<ProtosPartition>,
    Codebook<T>: Serialize<ProtosCodebook>,
    BlockVectorSet<T>: Serialize<ProtosVectorSet>,
    FS: FileSystem,
{
    // serializes partitions
    let partition_ids = serialize_partitions(db.partitions(), fs)?;
    // serializes partition centroids
    let partition_centroids_id =
        serialize_partition_centroids(&db.partitions, fs)?;
    // serializes codebooks
    let codebook_ids = serialize_codebooks(&db.codebooks, fs)?;
    // serializes attributes
    let attributes_log_id =
        serialize_attribute_table(&db.attribute_table, fs)?;
    // serializes the database
    let db = DatabaseSerialize {
        database: db,
        partition_ids,
        partition_centroids_id,
        codebook_ids,
        attributes_log_id,
    };
    let db = db.serialize()?;
    let mut f = fs.create_hashed_file()?;
    write_message(&db, &mut f)?;
    f.persist(PROTOBUF_EXTENSION)?;
    Ok(())
}

// Serializes partitions.
fn serialize_partitions<I, T, FS>(
    partitions: I,
    fs: &mut FS,
) -> Result<Vec<String>, Error>
where
    I: IntoIterator<Item = Partition<T>>,
    T: Clone,
    Partition<T>: Serialize<ProtosPartition>,
    FS: FileSystem,
{
    let mut partition_ids: Vec<String> = Vec::new();
    for partition in partitions {
        let partition_id = serialize_partition(&partition, fs)?;
        partition_ids.push(partition_id);
    }
    Ok(partition_ids)
}

// Serializes a partition.
fn serialize_partition<T, FS>(
    partition: &Partition<T>,
    fs: &mut FS,
) -> Result<String, Error>
where
    T: Clone,
    Partition<T>: Serialize<ProtosPartition>,
    FS: FileSystem,
{
    let partition = partition.serialize()?;
    let mut f = fs.create_hashed_file_in("partitions")?;
    write_message(&partition, &mut f)?;
    f.persist(PROTOBUF_EXTENSION)
}

// Serializes the partition centroids.
fn serialize_partition_centroids<T, VS, FS>(
    partitions: &Partitions<T, VS>,
    fs: &FS,
) -> Result<String, Error>
where
    BlockVectorSet<T>: Serialize<ProtosVectorSet>,
    FS: FileSystem,
{
    let partition_centroids: ProtosVectorSet =
        partitions.codebook.centroids.serialize()?;
    let mut f = fs.create_hashed_file_in("partitions")?;
    write_message(&partition_centroids, &mut f)?;
    f.persist(PROTOBUF_EXTENSION)
}

// Serializes codebooks.
fn serialize_codebooks<T, FS>(
    codebooks: &Vec<Codebook<T>>,
    fs: &mut FS,
) -> Result<Vec<String>, Error>
where
    Codebook<T>: Serialize<ProtosCodebook>,
    FS: FileSystem,
{
    let mut codebook_ids = Vec::with_capacity(codebooks.len());
    for codebook in codebooks {
        let codebook_id = serialize_codebook(codebook, fs)?;
        codebook_ids.push(codebook_id);
    }
    Ok(codebook_ids)
}

// Serializes a codebook.
fn serialize_codebook<T, FS>(
    codebook: &Codebook<T>,
    fs: &mut FS,
) -> Result<String, Error>
where
    Codebook<T>: Serialize<ProtosCodebook>,
    FS: FileSystem,
{
    let codebook = codebook.serialize()?;
    let mut f = fs.create_hashed_file_in("codebooks")?;
    write_message(&codebook, &mut f)?;
    f.persist(PROTOBUF_EXTENSION)
}

// Serializes an attribute table.
fn serialize_attribute_table<FS>(
    attribute_table: &HashMap<Uuid, Attributes>,
    fs: &mut FS,
) -> Result<String, Error>
where
    FS: FileSystem,
{
    let mut attributes_log = ProtosAttributesLog::new();
    for (id, attributes) in attribute_table.iter() {
        for (name, value) in attributes.iter() {
            let mut set_attribute = ProtosOperationSetAttribute::new();
            set_attribute.vector_id = Some(id.serialize()?).into();
            set_attribute.name = name.clone();
            set_attribute.value = Some(value.serialize()?).into();
            attributes_log.entries.push(set_attribute);
        }
    }
    let mut f = fs.create_hashed_file_in("attributes")?;
    write_message(&attributes_log, &mut f)?;
    f.persist(PROTOBUF_EXTENSION)
}

/// Serializable form of `Database`.
pub struct DatabaseSerialize<'a, T, VS>
where
    VS: VectorSet<T>,
{
    database: &'a Database<T, VS>,
    partition_ids: Vec<String>,
    partition_centroids_id: String,
    codebook_ids: Vec<String>,
    attributes_log_id: String,
}

impl<'a, T, VS> core::ops::Deref for DatabaseSerialize<'a, T, VS>
where
    VS: VectorSet<T>,
{
    type Target = Database<T, VS>;

    fn deref(&self) -> &Self::Target {
        self.database
    }
}

impl<'a, VS> Serialize<ProtosDatabase> for DatabaseSerialize<'a, f32, VS>
where
    VS: VectorSet<f32>,
{
    fn serialize(&self) -> Result<ProtosDatabase, Error> {
        let mut db = ProtosDatabase::new();
        db.vector_size = self.vector_size() as u32;
        db.num_partitions = self.num_partitions() as u32;
        db.num_divisions = self.num_divisions() as u32;
        db.num_codes = self.num_clusters() as u32;
        db.partition_ids = self.partition_ids.clone();
        db.partition_centroids_id = self.partition_centroids_id.clone();
        db.codebook_ids = self.codebook_ids.clone();
        db.attributes_log_id = self.attributes_log_id.clone();
        Ok(db)
    }
}

impl Serialize<ProtosPartition> for Partition<f32> {
    fn serialize(&self) -> Result<ProtosPartition, Error> {
        let mut partition = ProtosPartition::new();
        let n = self.num_vectors();
        let m = self.vector_size();
        let d = self.num_divisions();
        partition.vector_size = m as u32;
        partition.num_divisions = d as u32;
        partition.num_vectors = n as u32;
        partition.centroid.reserve(m);
        partition.centroid.extend_from_slice(&self.centroid[..]);
        partition.encoded_vectors.reserve(n);
        partition.vector_ids.reserve(n);
        for vi in 0..self.num_vectors() {
            let from = vi * d;
            let to = from + d;
            let mut encoded_vector = ProtosEncodedVector::new();
            encoded_vector.elements.reserve(d);
            encoded_vector.elements.extend(
                self.encoded_vectors[from..to].iter().map(|&c| c as u32),
            );
            partition.encoded_vectors.push(encoded_vector);
            partition.vector_ids.push(self.vector_ids[vi].serialize()?);
        }
        Ok(partition)
    }
}

impl Serialize<ProtosCodebook> for Codebook<f32> {
    fn serialize(&self) -> Result<ProtosCodebook, Error> {
        let mut codebook = ProtosCodebook::new();
        codebook.vector_size = self.centroids.vector_size() as u32;
        codebook.num_codes = self.centroids.len() as u32;
        codebook.codes.reserve(self.centroids.len());
        for ci in 0..self.centroids.len() {
            let centroid = self.centroids.get(ci);
            let mut code = ProtosVector::new();
            code.elements.reserve(centroid.len());
            code.elements.extend_from_slice(centroid);
            codebook.codes.push(code);
        }
        Ok(codebook)
    }
}

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
