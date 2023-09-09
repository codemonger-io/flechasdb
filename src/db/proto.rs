/// Serializes `Database` into a `protobuf::Message`.

use core::iter::IntoIterator;
use protobuf::CodedOutputStream;

use crate::error::Error;
use crate::io::{FileSystem, HashedFile};
use crate::kmeans::Codebook;
use crate::protos::database::{
    CodeVector as ProtosCodeVector,
    Codebook as ProtosCodebook,
    CodebookRef as ProtosCodebookRef,
    Database as ProtosDatabase,
    EncodedVector as ProtosEncodedVector,
    Partition as ProtosPartition,
    PartitionRef as ProtosPartitionRef,
};
use crate::protos::Serialize;
use crate::vector::VectorSet;
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
    FS: FileSystem,
{
    // serializes partitions
    let partition_ids = serialize_partitions(db.partitions(), fs)?;
    // serializes codebooks
    let codebook_ids = serialize_codebooks(&db.codebooks, fs)?;
    // serializes the database
    let db = DatabaseSerialize {
        database: db,
        partition_ids,
        codebook_ids,
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

// Writes a message to a stream.
fn write_message<M, W>(message: &M, out: &mut W) -> Result<(), Error>
where
    M: protobuf::Message,
    W: std::io::Write,
{
    let mut writer = CodedOutputStream::new(out);
    message.write_to(&mut writer)?;
    writer.flush()?;
    Ok(())
}

/// Serializable form of `Database`.
pub struct DatabaseSerialize<'a, T, VS>
where
    VS: VectorSet<T>,
{
    database: &'a Database<T, VS>,
    partition_ids: Vec<String>,
    codebook_ids: Vec<String>,
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
        db.partition_refs.reserve(self.partition_ids.len());
        for (pi, id) in self.partition_ids.iter().enumerate() {
            let mut partition_ref = ProtosPartitionRef::new();
            partition_ref.id = id.clone();
            let centroid = self.partitions.codebook.centroids.get(pi);
            partition_ref.centroid.reserve(self.vector_size);
            partition_ref.centroid.extend_from_slice(centroid);
            db.partition_refs.push(partition_ref);
        }
        db.codebook_refs.reserve(self.codebook_ids.len());
        db.codebook_refs.extend(self.codebook_ids
            .iter()
            .map(|id| {
                let mut codebook_ref = ProtosCodebookRef::new();
                codebook_ref.id = id.clone();
                codebook_ref
            }),
        );
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
            encoded_vector.index = vi as u32;
            encoded_vector.code_indices.reserve(d);
            encoded_vector.code_indices.extend(
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
            let mut code = ProtosCodeVector::new();
            code.elements.reserve(centroid.len());
            code.elements.extend_from_slice(centroid);
            codebook.codes.push(code);
        }
        Ok(codebook)
    }
}
