//! Asynchronous stored database.

use async_trait::async_trait;
use core::borrow::Borrow;
use core::hash::Hash;
use core::marker::{PhantomData, Send, Sync};
use core::num::NonZeroUsize;
use futures::future::try_join_all;
use std::collections::hash_map::{Entry as HashMapEntry};
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard, OnceCell};
use uuid::Uuid;

use crate::db::{AttributeValue, AttributeTable, Attributes};
use crate::error::Error;
use crate::protos::Deserialize;
use crate::protos::database::{
    AttributesLog as ProtosAttributesLog,
    Database as ProtosDatabase,
    Partition as ProtosPartition,
    VectorSet as ProtosVectorSet,
};
use crate::slice::AsSlice;
use crate::vector::BlockVectorSet;

use super::io::{FileSystem, HashedFileIn};
use super::proto::read_message;

pub mod get_attribute;
pub mod query;
pub use query::{Query, QueryEvent, QueryResult};

/// Extension for Protocol Buffers files.
pub const PROTOBUF_EXTENSION: &str = "binpb";

/// Asynchronous database associated with an asynchronous file system.
pub struct Database<T, FS>
where
    T: Send,
    FS: Send,
{
    fs: FS,
    vector_size: usize,
    num_partitions: usize,
    num_divisions: usize,
    num_codes: usize,
    partition_ids: Vec<String>,
    partitions: Vec<OnceCell<Partition<T>>>,
    partition_centroids_id: String,
    partition_centroids: OnceCell<BlockVectorSet<T>>,
    codebook_ids: Vec<String>,
    codebooks: OnceCell<Vec<BlockVectorSet<T>>>,
    attributes_log_ids: Vec<String>,
    attributes_log_load_flags: Vec<OnceCell<bool>>,
    attribute_table: Mutex<AttributeTable>,
}

impl<T, FS> Database<T, FS>
where
    T: Send,
    FS: Send,
{
    /// Returns the vector size.
    pub const fn vector_size(&self) -> usize {
        self.vector_size
    }

    /// Returns the number of partitions.
    pub const fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Returns the number of divisions.
    pub const fn num_divisions(&self) -> usize {
        self.num_divisions
    }

    /// Returns the number of codes.
    pub const fn num_codes(&self) -> usize {
        self.num_codes
    }

    // Returns the attribute value.
    //
    // Supposes the attributes log of the partition where a given vector
    // belongs to has been loaded.
    //
    // A returned value holds the lock of the attribute table, so you have to
    // drop it as soon as it becomes unnecessary.
    async fn get_attribute_internal<'a, K>(
        &'a self,
        uuid: &Uuid,
        key: &K,
    ) -> Result<Option<AttributeValueRef<'a>>, Error>
    where
        String: Borrow<K>,
        K: Hash + Eq + ?Sized,
    {
        let attribute_table = self.attribute_table.lock().await;
        let attributes = MutexGuard::try_map(
            attribute_table,
            |tbl| tbl.get_mut(uuid),
        ).or(Err(Error::InvalidArgs(format!("no such vector: {}", uuid))))?;
        match MappedMutexGuard::try_map(
            attributes,
            |attrs| attrs.get_mut(&key),
        ) {
            Ok(value) => Ok(Some(value)),
            Err(_) => Ok(None),
        }
    }
}

// Reference to an attribute value.
type AttributeValueRef<'a> = MappedMutexGuard<'a, AttributeValue>;

impl<'db, T, FS> Database<T, FS>
where
    T: Send,
    FS: Send,
    Self: 'db + LoadPartitionCentroids<'db, T>,
{
    /// Queries k-nearest neighbors of a given vector.
    pub fn query<'v, V>(
        &'db self,
        v: &'v V,
        k: NonZeroUsize,
        nprobe: NonZeroUsize,
    ) -> Query<'db, 'v, T, FS, V, impl FnMut(QueryEvent)>
    where
        V: AsSlice<T> + Send + ?Sized,
    {
        self.query_with_events(v, k, nprobe, |_| {})
    }

    /// Queries k-nearest neighbors of a given vector.
    pub fn query_with_events<'v, V, EV>(
        &'db self,
        v: &'v V,
        k: NonZeroUsize,
        nprobe: NonZeroUsize,
        event_handler: EV,
    ) -> Query<'db, 'v, T, FS, V, EV>
    where
        V: AsSlice<T> + Send + ?Sized,
        EV: FnMut(QueryEvent),
    {
        Query::new(self, v, k, nprobe, event_handler)
    }
}

/// Partition.
pub struct Partition<T> {
    _t: PhantomData<T>,
    encoded_vectors: BlockVectorSet<u32>,
    vector_ids: Vec<Uuid>,
}

impl<T> Partition<T> {
    const fn num_divisions(&self) -> usize {
        self.encoded_vectors.vector_size()
    }

    fn num_vectors(&self) -> usize {
        self.encoded_vectors.len()
    }

    // Panics if the index is out of bounds.
    fn get_encoded_vector<'a>(&'a self, index: usize) -> &'a [u32] {
        self.encoded_vectors.get(index)
    }

    // Panics if the index is out of bounds.
    fn get_vector_id<'a>(&'a self, index: usize) -> &'a Uuid {
        &self.vector_ids[index]
    }
}

/// Capability of loading a database.
///
/// Supposed to be specialized for a specific [`Database`].
#[async_trait]
pub trait LoadDatabase<T, FS> {
    /// Loads a database.
    async fn load_database<P>(fs: FS, path: P) -> Result<Database<T, FS>, Error>
    where
        T: Send,
        FS: Send,
        P: Into<String> + Send;
}

/// Capability of loading a partition centroids.
///
/// Supposed to be specialized for a specific [`Database`].
#[async_trait]
pub trait LoadPartitionCentroids<'db, T> {
    /// Loads the partition centroids of the database.
    async fn load_partition_centroids(
        &'db self,
    ) -> Result<&'db BlockVectorSet<T>, Error>;
}

/// Capability of loading a single codebook.
///
/// Supposed to be specialized for a specific [`Database`].
#[async_trait]
pub trait LoadCodebook<T> {
    /// Loads a specified codebook of the database.
    ///
    /// Fails if `index` is out of bounds.
    async fn load_codebook(
        &self,
        index: usize,
    ) -> Result<BlockVectorSet<T>, Error>;
}

/// Capability of loading a single partition.
///
/// Supposed to be specialized for a specific [`Database`].
#[async_trait]
pub trait LoadPartition<'db, T> {
    /// Loads a specified partition of the database.
    ///
    /// Fails if `index` is out of bounds.
    async fn load_partition(
        &'db self,
        index: usize,
    ) -> Result<&'db Partition<T>, Error>;
}

/// Capability of loading the attributes log of a partition.
///
/// Supposed to be specialized for a specific [`Database`].
#[async_trait]
pub trait LoadAttributesLog<'db> {
    /// Loads the attributes log of a partition in the database.
    ///
    /// Fails if `index` is out of bounds.
    async fn load_attributes_log(&'db self, index: usize) -> Result<(), Error>;
}

impl<'db, T, FS> Database<T, FS>
where
    T: Send,
    FS: Send,
    Self: LoadCodebook<T>
{
    // Loads all the codebooks if not loaded.
    async fn load_codebooks(
        &'db self,
    ) -> Result<&'db Vec<BlockVectorSet<T>>, Error> {
        self.codebooks.get_or_try_init(|| try_join_all(
            (0..self.num_divisions()).map(|i| self.load_codebook(i)),
        )).await
    }
}

#[async_trait]
impl<'db, T, FS> LoadAttributesLog<'db> for Database<T, FS>
where
    T: Send + Sync,
    FS: FileSystem + Send + Sync,
    Self: LoadPartition<'db, T> + Sync,
{
    async fn load_attributes_log(&'db self, index: usize) -> Result<(), Error> {
        if index >= self.num_partitions() {
            return Err(Error::InvalidArgs(format!(
                "partition index {} must be < {}",
                index,
                self.num_partitions(),
            )));
        }
        self.attributes_log_load_flags[index].get_or_try_init(|| async move {
            let partition = self.load_partition(index).await?;
            let id = &self.attributes_log_ids[index];
            let mut f = self.fs.open_hashed_file(format!(
                "attributes/{}.{}",
                id,
                PROTOBUF_EXTENSION,
            )).await?;
            let attributes_log: ProtosAttributesLog =
                read_message(&mut f).await?;
            f.verify().await?;
            if attributes_log.partition_id != self.partition_ids[index] {
                return Err(Error::InvalidData(format!(
                    "inconsistent partition IDs: {} vs {}",
                    attributes_log.partition_id,
                    self.partition_ids[index],
                )));
            }
            let mut attribute_table = self.attribute_table.lock().await;
            for (i, entry) in attributes_log.entries.into_iter().enumerate() {
                let vector_id = entry.vector_id
                    .into_option()
                    .ok_or(Error::InvalidData(format!(
                        "attributes log[{}, {}]: missing vector ID",
                        index,
                        i,
                    )))?
                    .deserialize()?;
                let value = entry.value
                    .into_option()
                    .ok_or(Error::InvalidData(format!(
                        "attributes log[{}, {}]: missing value",
                        index,
                        i,
                    )))?
                    .deserialize()?;
                match attribute_table.entry(vector_id) {
                    HashMapEntry::Occupied(slot) => {
                        match slot.into_mut().entry(entry.name) {
                            HashMapEntry::Occupied(slot) => {
                                *slot.into_mut() = value;
                            },
                            HashMapEntry::Vacant(slot) => {
                                slot.insert(value);
                            },
                        };
                    },
                    HashMapEntry::Vacant(slot) => {
                        slot.insert(Attributes::from([(entry.name, value)]));
                    },
                };
            }
            // defaults to empty attributes so that get_attribute won't fail
            // for an existing vector without attributes.
            for vector_id in partition.vector_ids.iter() {
                attribute_table
                    .entry(vector_id.clone())
                    .or_insert_with(Attributes::new);
            }
            Ok(true)
        }).await?;
        Ok(())
    }
}

mod f32impl {
    use super::*;

#[async_trait]
    impl<FS> LoadDatabase<f32, FS> for Database<f32, FS>
    where
        for<'a> FS: 'a + FileSystem + Send + Sync,
    {
        async fn load_database<P>(
            fs: FS,
            path: P,
        ) -> Result<Database<f32, FS>, Error>
        where
            P: Into<String> + Send,
        {
            let mut f = fs.open_hashed_file(path).await?;
            let db: ProtosDatabase = read_message(&mut f).await?;
            f.verify().await?;
            let vector_size = db.vector_size as usize;
            let num_partitions = db.num_partitions as usize;
            let num_divisions = db.num_divisions as usize;
            let num_codes = db.num_codes as usize;
            if vector_size == 0 {
                return Err(Error::InvalidData(format!("vector_size is zero")));
            }
            if num_divisions == 0 {
                return Err(Error::InvalidData(
                    format!("num_divisions is zero"),
                ));
            }
            if num_partitions == 0 {
                return Err(Error::InvalidData(
                    format!("num_partitions is zero"),
                ));
            }
            if num_codes == 0 {
                return Err(Error::InvalidData(format!("num_codes is zero")));
            }
            if vector_size % num_divisions != 0 {
                return Err(Error::InvalidData(format!(
                    "vector_size {} is not multiple of num_divisions {}",
                    vector_size,
                    num_divisions,
                )));
            }
            if num_partitions != db.partition_ids.len() {
                return Err(Error::InvalidData(format!(
                    "num_partitions {} and partition_ids.len() {} do not match",
                    num_partitions,
                    db.partition_ids.len(),
                )));
            }
            if num_divisions != db.codebook_ids.len() {
                return Err(Error::InvalidData(format!(
                    "num_divisions {} and codebook_ids.len() {} do not match",
                    num_divisions,
                    db.codebook_ids.len(),
                )));
            }
            let mut partitions = Vec::with_capacity(num_partitions);
            partitions.resize_with(num_partitions, OnceCell::new);
            let mut attributes_log_load_flags =
                Vec::with_capacity(num_partitions);
            attributes_log_load_flags.resize_with(
                num_partitions,
                OnceCell::new,
            );
            Ok(
                Database {
                    fs,
                    vector_size,
                    num_partitions,
                    num_divisions,
                    num_codes,
                    partition_ids: db.partition_ids,
                    partitions,
                    partition_centroids_id: db.partition_centroids_id,
                    partition_centroids: OnceCell::new(),
                    codebook_ids: db.codebook_ids,
                    codebooks: OnceCell::new(),
                    attributes_log_ids: db.attributes_log_ids,
                    attributes_log_load_flags,
                    attribute_table: Mutex::new(AttributeTable::new()),
                }
            )
        }
    }

    #[async_trait]
    impl<'db, FS> LoadPartitionCentroids<'db, f32> for Database<f32, FS>
    where
        FS: FileSystem + Send + Sync,
        Self: 'db,
    {
        async fn load_partition_centroids(
            &'db self,
        ) -> Result<&'db BlockVectorSet<f32>, Error> {
            self.partition_centroids.get_or_try_init(|| async move {
                let mut f = self.fs.open_hashed_file(format!(
                    "partitions/{}.{}",
                    self.partition_centroids_id,
                    PROTOBUF_EXTENSION,
                )).await?;
                let partition_centroids: ProtosVectorSet =
                    read_message(&mut f).await?;
                f.verify().await?;
                let partition_centroids: BlockVectorSet<f32> =
                    partition_centroids.deserialize()?;
                Ok(partition_centroids)
            }).await
        }
    }

    #[async_trait]
    impl<FS> LoadCodebook<f32> for Database<f32, FS>
    where
        FS: FileSystem + Send + Sync,
    {
        async fn load_codebook(
            &self,
            index: usize,
        ) -> Result<BlockVectorSet<f32>, Error> {
            if index >= self.num_divisions() {
                return Err(Error::InvalidArgs(format!(
                    "codebook index {} must be < {}",
                    index,
                    self.num_divisions(),
                )));
            }
            let mut f = self.fs.open_hashed_file(format!(
                "codebooks/{}.{}",
                &self.codebook_ids[index],
                PROTOBUF_EXTENSION,
            )).await?;
            let codebook: ProtosVectorSet = read_message(&mut f).await?;
            f.verify().await?;
            let codebook: BlockVectorSet<f32> = codebook.deserialize()?;
            Ok(codebook)
        }
    }

    #[async_trait]
    impl<'db, FS> LoadPartition<'db, f32> for Database<f32, FS>
    where
        FS: FileSystem + Send + Sync,
        Self: 'db,
    {
        async fn load_partition(
            &'db self,
            index: usize,
        ) -> Result<&'db Partition<f32>, Error> {
            if index >= self.num_partitions() {
                return Err(Error::InvalidArgs(format!(
                    "partition index {} must be < {}",
                    index,
                    self.num_partitions(),
                )));
            }
            self.partitions[index].get_or_try_init(|| async move {
                let id = &self.partition_ids[index];
                let mut f = self.fs.open_hashed_file(format!(
                    "partitions/{}.{}",
                    id,
                    PROTOBUF_EXTENSION,
                )).await?;
                let partition: ProtosPartition = read_message(&mut f).await?;
                f.verify().await?;
                let vector_size = partition.vector_size as usize;
                let num_divisions = partition.num_divisions as usize;
                let encoded_vectors: BlockVectorSet<u32> = partition.encoded_vectors
                    .into_option()
                    .ok_or(Error::InvalidData(format!(
                        "missing encoded vectors for partition: {}",
                        id,
                    )))?
                    .deserialize()?;
                if vector_size != self.vector_size() {
                    return Err(Error::InvalidData(format!(
                        "inconsistent vector size: expected {} but got {}",
                        self.vector_size(),
                        vector_size,
                    )));
                }
                if num_divisions != self.num_divisions() {
                    return Err(Error::InvalidData(format!(
                        "inconsistent # of divisions: expected {} but got {}",
                        self.num_divisions(),
                        num_divisions,
                    )));
                }
                if encoded_vectors.len() != partition.vector_ids.len() {
                    return Err(Error::InvalidData(format!(
                        "inconsistent # of vectors: {} and {}",
                        encoded_vectors.len(),
                        partition.vector_ids.len(),
                    )));
                }
                let vector_ids: Vec<Uuid> = partition.vector_ids
                    .into_iter()
                    .map(|id| id.deserialize().unwrap())
                    .collect();
                Ok(Partition {
                    _t: std::marker::PhantomData,
                    encoded_vectors,
                    vector_ids,
                })
            }).await
        }
    }
}
