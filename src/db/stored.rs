//! Defines a stored database.

use core::borrow::Borrow;
use core::cell::{OnceCell, Ref, RefCell, RefMut};
use core::hash::Hash;
use core::num::NonZeroUsize;
use std::collections::HashMap;
use std::collections::hash_map::{Entry as HashMapEntry};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use crate::db::types::{Attributes, AttributeValue};
use crate::error::Error;
use crate::io::{FileSystem, HashedFileIn};
use crate::kmeans::Scalar;
use crate::linalg::{dot, subtract};
use crate::protos::database::{
    AttributesLog as ProtosAttributesLog,
    Codebook as ProtosCodebook,
    Database as ProtosDatabase,
    Partition as ProtosPartition,
    VectorSet as ProtosVectorSet,
};
use crate::protos::{Deserialize, read_message};
use crate::slice::AsSlice;
use crate::vector::BlockVectorSet;

pub const PROTOBUF_EXTENSION: &str = "binpb";

/// Database store.
pub struct DatabaseStore<T, FS> {
    _t: std::marker::PhantomData<T>,
    _fs: std::marker::PhantomData<FS>,
}

/// Interface to load a database.
///
/// Supposed to be implemented by `DatabaseStore`.
pub trait LoadDatabase<T, FS> {
    /// Loads a database.
    fn load_database<P>(fs: FS, path: P) -> Result<Database<T, FS>, Error>
    where
        P: AsRef<Path>;
}

/// Stored database.
pub struct Database<T, FS> {
    fs: FS,
    vector_size: usize,
    num_partitions: usize,
    num_divisions: usize,
    num_codes: usize,
    partition_ids: Vec<String>,
    partitions: RefCell<Vec<Option<Partition<T>>>>,
    partition_centroids_id: String,
    partition_centroids: OnceCell<BlockVectorSet<T>>,
    codebook_ids: Vec<String>,
    codebooks: RefCell<Option<Vec<Codebook<T>>>>,
    attributes_log_ids: Vec<String>,
    attributes_log_load_flags: RefCell<Vec<bool>>,
    attribute_table: RefCell<Option<AttributeTable>>,
}

impl<T, FS> Database<T, FS>
where
    FS: FileSystem,
{
    /// Returns the vector size.
    pub fn vector_size(&self) -> usize {
        self.vector_size
    }

    /// Returns the number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.num_partitions
    }

    /// Returns the number of subvector divisions.
    pub fn num_divisions(&self) -> usize {
        self.num_divisions
    }

    /// Returns the number of codes in each codebook.
    pub fn num_codes(&self) -> usize {
        self.num_codes
    }

    /// Returns the subvector size.
    pub fn subvector_size(&self) -> usize {
        self.vector_size / self.num_divisions
    }

    /// Returns the ID of a partition.
    ///
    /// `None` if `index` ≥ `num_partitions`.
    pub fn get_partition_id(&self, index: usize) -> Option<&String> {
        self.partition_ids.get(index)
    }

    /// Returns a code vector in a specified division.
    ///
    /// `None` if `index` ≥ `num_divisons`.
    pub fn get_codebook_id(&self, index: usize) -> Option<&String> {
        self.codebook_ids.get(index)
    }
}

impl<T, FS> Database<T, FS>
where
    FS: FileSystem,
    Self: LoadPartition<T>,
{
    /// Returns an attribute value of a given vector.
    ///
    /// The first call to this function will take longer because it loads all
    /// the attributes.
    /// If you want to get attributes of your query results, please use
    /// `get_attribute_of` instead.
    ///
    /// `None` if the vector exists but no value is associated with `key`.
    ///
    /// Fails if no vector is associated with `id`.
    pub fn get_attribute<K>(
        &self,
        id: &Uuid,
        key: &K,
    ) -> Result<Option<AttributeValueRef>, Error>
    where
        String: Borrow<K>,
        K: Hash + Eq + ?Sized,
    {
        if self.attribute_table.borrow().is_none() {
            self.load_attribute_table()?;
        }
        self.get_attribute_no_loading(id, key)
    }

    /// Returns an attribute value of a given vector.
    ///
    /// This function will be more efficient if you have a query result.
    ///
    /// `None` if no value is associated with `key`.
    ///
    /// Fails if a query result is not from this database.
    pub fn get_attribute_of<K>(
        &self,
        query_result: &QueryResult<T>,
        key: &K,
    ) -> Result<Option<AttributeValueRef>, Error>
    where
        String: Borrow<K>,
        K: Hash + Eq + ?Sized,
    {
        let partition_index = query_result.partition_index;
        if self.partition_ids[partition_index] != query_result.partition_id {
            return Err(Error::InvalidArgs(format!(
                "incompatible patition: expected {} but got {}",
                self.partition_ids[partition_index],
                query_result.partition_id,
            )));
        }
        if self.attributes_log_ids[partition_index] != query_result.attributes_log_id {
            return Err(Error::InvalidArgs(format!(
                "incompatible attributes log: expected {} but got {}",
                self.attributes_log_ids[partition_index],
                query_result.attributes_log_id,
            )));
        }
        self.load_attributes_log(partition_index)?;
        self.get_attribute_no_loading(&query_result.vector_id, key)
    }

    fn get_attribute_no_loading<K>(
        &self,
        id: &Uuid,
        key: &K,
    ) -> Result<Option<AttributeValueRef>, Error>
    where
        String: Borrow<K>,
        K: Hash + Eq + ?Sized,
    {
        let attribute_table = Ref::filter_map(
            self.attribute_table.borrow(),
            |tbl| tbl.as_ref(),
        ).expect("attribute table must be loaded");
        let attributes = Ref::filter_map(
            attribute_table,
            |tbl| tbl.get(id),
        ).or(Err(Error::InvalidArgs(format!("no such vector ID: {}", id))))?;
        match Ref::filter_map(attributes, |attrs| attrs.get(key)) {
            Ok(value) => Ok(Some(value)),
            Err(_) => Ok(None),
        }
    }

    fn load_attribute_table(&self) -> Result<(), Error> {
        for pi in 0..self.num_partitions() {
            self.load_attributes_log(pi)?;
        }
        Ok(())
    }

    // Loads the attributes log of a specified partition if it is not loaded
    // yet.
    //
    // This function also loads the partition to list all the vector IDs in
    // the partition.
    fn load_attributes_log(&self, partition_index: usize) -> Result<(), Error> {
        if self.attributes_log_load_flags.borrow()[partition_index] {
            return Ok(());
        }
        let partition = self.get_partition(partition_index)?;
        let mut path = PathBuf::from("attributes");
        path.push(&self.attributes_log_ids[partition_index]);
        path.set_extension(PROTOBUF_EXTENSION);
        let mut f = self.fs.open_hashed_file(path)?;
        let attributes_log: ProtosAttributesLog = read_message(&mut f)?;
        if attributes_log.partition_id != self.partition_ids[partition_index] {
            return Err(Error::InvalidData(format!(
                "inconsistent partition IDs: {} vs {}",
                attributes_log.partition_id,
                self.partition_ids[partition_index],
            )));
        }
        if self.attribute_table.borrow().is_none() {
            self.attribute_table.replace(Some(AttributeTable::new()));
        }
        let mut attribute_table = RefMut::filter_map(
            self.attribute_table.borrow_mut(),
            |tbl| tbl.as_mut(),
        ).expect("attribute table must exist");
        for (i, entry) in attributes_log.entries.into_iter().enumerate() {
            let vector_id = entry.vector_id
                .into_option()
                .ok_or(Error::InvalidData(format!(
                    "attributes log[{}, {}]: missing vector ID",
                    partition_index,
                    i,
                )))?
                .deserialize()?;
            let value = entry.value
                .into_option()
                .ok_or(Error::InvalidData(format!(
                    "attributes log[{}, {}]: missing value",
                    partition_index,
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
        // defaults to empty attributes so that
        // get_attribute won't fail for an existing vector without attributes.
        for vector_id in partition.vector_ids.iter() {
            attribute_table
                .entry(vector_id.clone())
                .or_insert_with(Attributes::new);
        }
        self.attributes_log_load_flags.borrow_mut()[partition_index] = true;
        Ok(())
    }

    // Obtains a specified partition.
    //
    // Lazily loads the partition if it is not loaded yet.
    //
    // Fails if:
    // - `index` exceeds the number of partitions
    // - there is any problem on the partition data
    fn get_partition(
        &self,
        index: usize,
    ) -> Result<PartitionRef<'_, T>, Error> {
        if index >= self.num_partitions() {
            return Err(Error::InvalidArgs(format!(
                "partition index out of bounds: {}",
                index,
            )));
        }
        if self.partitions.borrow()[index].is_none() {
            self.partitions.borrow_mut()[index] =
                Some(self.load_partition(index)?);
        }
        let partition =
            Ref:: filter_map(
                self.partitions.borrow(),
                |partitions| partitions[index].as_ref(),
            )
            .or(Err(Error::InvalidData(
                "partition must be loaded".to_string(),
            )))
            .unwrap();
        Ok(partition)
    }
}

// Reference type of a partition.
type PartitionRef<'a, T> = Ref<'a, Partition<T>>;

/// Reference type of an attribute value.
pub type AttributeValueRef<'a> = Ref<'a, AttributeValue>;

impl<FS> LoadPartition<f32> for Database<f32, FS>
where
    FS: FileSystem,
{
    /// Loads a partition.
    ///
    /// Loads a Protocol Buffers message (`p`) from the file system.
    ///
    /// Fails if:
    /// - `index` exceeds the number of partitions.
    /// - `self.vector_size` and `p.vector_size` do not match
    /// - `self.num_divisions` and `p.num_divisions` do not match
    /// - `p.num_vectors` and `p.encoded_vectors.len()` do not match
    /// - `p.num_vectors` and `p.vector_ids.len()` do not match
    /// - `p.num_divisions` and encoded vector length do not match
    fn load_partition(
        &self,
        index: usize,
    ) -> Result<Partition<f32>, Error> {
        if index >= self.num_partitions {
            return Err(Error::InvalidArgs(format!(
                "index {} exceeds the number of partitions {}",
                index,
                self.num_partitions,
            )));
        }
        let mut path = PathBuf::from("partitions");
        path.push(self.get_partition_id(index).unwrap());
        path.set_extension(PROTOBUF_EXTENSION);
        let mut f = self.fs.open_hashed_file(path)?;
        let partition: ProtosPartition = read_message(&mut f)?;
        f.verify()?;
        let vector_size = partition.vector_size as usize;
        let num_divisions = partition.num_divisions as usize;
        let encoded_vectors: BlockVectorSet<u32> = partition.encoded_vectors
                .into_option()
                .ok_or(Error::InvalidData(
                    "missing encoded vectors".to_string(),
                ))?
                .deserialize()?;
        if vector_size != self.vector_size() {
            return Err(Error::InvalidData(format!(
                "vector_size {} and partition.vector_size {} do not match",
                self.vector_size(),
                vector_size,
            )));
        }
        if num_divisions != self.num_divisions() {
            return Err(Error::InvalidData(format!(
                "num_divisions {} and partition.num_divisions {} do not match",
                self.num_divisions(),
                num_divisions,
            )));
        }
        if encoded_vectors.len() != partition.vector_ids.len() {
            return Err(Error::InvalidData(format!(
                "number of vector IDs is inconsistent: exptected {} but got {}",
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
    }
}

impl<FS> LoadCodebook<f32> for Database<f32, FS>
where
    FS: FileSystem,
{
    /// Loads a codebook.
    ///
    /// Suppose `c` represents a Protocol Buffers message for a codebook.
    /// Fails if:
    /// - `index` exceeds the number of codebooks.
    /// - `self.subvector_size()` and `c.vector_size` do not match.
    /// - `c.num_codes` and `c.codes.len()` do not match.
    /// - `c.vector_size` and the code vector length do not match.
    fn load_codebook(&self, index: usize) -> Result<Codebook<f32>, Error>
    where
        FS: FileSystem,
    {
        if index >= self.num_divisions() {
            return Err(Error::InvalidArgs(format!(
                "index {} exceeds the number of codebooks {}",
                index,
                self.num_divisions(),
            )));
        }
        let path = PathBuf::from("codebooks")
            .join(self.get_codebook_id(index).unwrap())
            .with_extension(PROTOBUF_EXTENSION);
        let mut f = self.fs.open_hashed_file(path)?;
        let codebook: ProtosCodebook = read_message(&mut f)?;
        f.verify()?;
        let vector_size = codebook.vector_size as usize;
        let num_codes = codebook.num_codes as usize;
        if vector_size != self.subvector_size() {
            return Err(Error::InvalidData(format!(
                "vector_size {} and subvector length {} do not match",
                vector_size,
                self.subvector_size(),
            )));
        }
        if num_codes != codebook.codes.len() {
            return Err(Error::InvalidData(format!(
                "number of codes is inconsistent: expected {} but got {}",
                num_codes,
                codebook.codes.len(),
            )));
        }
        // loads codes
        let mut codes: Vec<f32> = Vec::with_capacity(num_codes * vector_size);
        for code_vector in codebook.codes.into_iter() {
            if vector_size != code_vector.elements.len() {
                return Err(Error::InvalidData(format!(
                    "vector size is inconsitent: expected {} but got {}",
                    vector_size,
                    code_vector.elements.len(),
                )));
            }
            codes.extend(code_vector.elements);
        }
        Ok(Codebook {
            vector_size,
            num_codes,
            codes,
        })
    }
}

impl<FS> LoadPartitionCentroids<f32> for Database<f32, FS>
where
    FS: FileSystem,
{
    fn load_partition_centroids(&self) -> Result<BlockVectorSet<f32>, Error> {
        let mut path = PathBuf::from("partitions");
        path.push(&self.partition_centroids_id);
        path.set_extension(PROTOBUF_EXTENSION);
        let mut f = self.fs.open_hashed_file(path)?;
        let partition_centroids: ProtosVectorSet = read_message(&mut f)?;
        let partition_centroids: BlockVectorSet<f32> =
            partition_centroids.deserialize()?;
        if partition_centroids.vector_size() != self.vector_size() {
            return Err(Error::InvalidData(format!(
                "partition centroids vector size mismatch: expected {}, got {}",
                self.vector_size(),
                partition_centroids.vector_size(),
            )));
        }
        if partition_centroids.len() != self.num_partitions() {
            return Err(Error::InvalidData(format!(
                "partition centroids data length mismatch: expected {}, got {}",
                self.num_partitions(),
                partition_centroids.len(),
            )));
        }
        Ok(partition_centroids)
    }
}

impl<T, FS> Database<T, FS>
where
    T: Scalar,
    FS: FileSystem,
    Self: LoadPartition<T> + LoadCodebook<T> + LoadPartitionCentroids<T>,
{
    /// Queries k-nearest neighbors (k-NN) of a given vector.
    ///
    /// The first call to this function will take longer because it lazily
    /// loads partition centroids, and codebooks.
    pub fn query<V, EventHandler>(
        &self,
        v: &V,
        k: NonZeroUsize,
        nprobe: NonZeroUsize,
        mut event_handler: Option<EventHandler>,
    ) -> Result<Vec<QueryResult<T>>, Error>
    where
        V: AsSlice<T>,
        EventHandler: FnMut(DatabaseQueryEvent) -> (),
    {
        macro_rules! event {
            ($event:expr) => {
                event_handler.iter_mut().for_each(|f| f($event))
            };
        }
        event!(DatabaseQueryEvent::StartingQueryInitialization);
        if self.partition_centroids.get().is_none() {
            // lazily loads partition centroids
            self.partition_centroids
                .set(self.load_partition_centroids()?)
                .unwrap();
        }
        if self.codebooks.borrow().is_none() {
            // loads codebooks if not loaded yet.
            let mut codebooks: Vec<Codebook<T>> =
                Vec::with_capacity(self.num_divisions());
            for di in 0..self.num_divisions() {
                codebooks.push(self.load_codebook(di)?);
            }
            self.codebooks.replace(Some(codebooks));
        }
        event!(DatabaseQueryEvent::FinishedQueryInitialization);
        event!(DatabaseQueryEvent::StartingPartitionSelection);
        let v = v.as_slice();
        let queries = self.query_partitions(v, nprobe)?;
        event!(DatabaseQueryEvent::FinishedPartitionSelection);
        let mut all_results: Vec<QueryResult<T>> = Vec::new();
        for query in queries.into_iter() {
            event!(DatabaseQueryEvent::StartingPartitionQuery(
                query.partition_index,
            ));
            let results = query.execute()?;
            all_results.extend(results);
            event!(DatabaseQueryEvent::FinishedPartitionQuery(
                query.partition_index,
            ));
        }
        event!(DatabaseQueryEvent::StartingResultSelection);
        all_results.sort_by(|lhs, rhs| {
            lhs.squared_distance.partial_cmp(&rhs.squared_distance).unwrap()
        });
        all_results.truncate(k.get());
        event!(DatabaseQueryEvent::FinishedResultSelection);
        Ok(all_results)
    }

    // Queries partitions closest to a given vector.
    //
    // Panics if the partition centroids are not loaded.
    fn query_partitions<'a>(
        &'a self,
        v: &[T],
        nprobe: NonZeroUsize,
    ) -> Result<Vec<PartitionQuery<'a, T, FS>>, Error> {
        let nprobe = nprobe.get();
        let num_partitions = self.num_partitions();
        if nprobe > num_partitions {
            return Err(Error::InvalidArgs(format!(
                "nprobe {} exceeds the number of partitions {}",
                nprobe,
                num_partitions,
            )));
        }
        let partition_centroids = self.partition_centroids.get()
            .expect("partition centroids must be loaded");
        // localizes vectors and calculates distances
        let mut distances: Vec<(usize, Vec<T>, T)> =
            Vec::with_capacity(num_partitions);
        for pi in 0..num_partitions {
            let mut localized: Vec<T> = Vec::with_capacity(self.vector_size());
            unsafe {
                localized.set_len(self.vector_size());
            }
            let centroid = partition_centroids.get(pi);
            subtract(v, &centroid, &mut localized[..]);
            let distance = dot(&localized[..], &localized[..]);
            distances.push((pi, localized, distance));
        }
        // chooses `nprobes` shortest distances.
        distances.sort_by(|lhs, rhs| lhs.2.partial_cmp(&rhs.2).unwrap());
        distances.truncate(nprobe);
        // makes queries.
        let queries = distances
            .into_iter()
            .map(|(pi, localized, _)| PartitionQuery {
                database: self,
                codebooks: Ref::map(
                    self.codebooks.borrow(),
                    |cb| cb.as_ref().unwrap(),
                ),
                partition_index: pi,
                localized,
            })
            .collect();
        Ok(queries)
    }
}

impl<FS> LoadDatabase<f32, FS> for DatabaseStore<f32, FS>
where
    FS: FileSystem,
{
    /// Loads a database.
    ///
    /// Fails if:
    /// - `vector_size` is zero
    /// - `num_divisions` is zero
    /// - `num_partitions` is zero
    /// - `num_codes` is zero
    /// - `vector_size` is not a multiple of `num_divisions`
    /// - `num_partitions` and `partitions_refs.len()` do not match
    /// - `vector_size` and centroid size do not match
    /// - `num_divisions` and `codebook_refs.len()` do not match
    fn load_database<P>(fs: FS, path: P) -> Result<Database<f32, FS>, Error>
    where
        P: AsRef<Path>,
    {
        let mut f = fs.open_hashed_file(path)?;
        let db: ProtosDatabase = read_message(&mut f)?;
        f.verify()?;
        let vector_size = db.vector_size as usize;
        let num_partitions = db.num_partitions as usize;
        let num_divisions = db.num_divisions as usize;
        let num_codes = db.num_codes as usize;
        if vector_size == 0 {
            return Err(Error::InvalidData(format!("vector_size is zero")));
        }
        if num_divisions == 0 {
            return Err(Error::InvalidData(format!("num_divisions is zero")));
        }
        if num_partitions == 0 {
            return Err(Error::InvalidData(format!("num_partitions is zero")));
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
                db.num_partitions,
                db.partition_ids.len(),
            )));
        }
        if num_divisions != db.codebook_ids.len() {
            return Err(Error::InvalidData(format!(
                "num_divisions {} and codebook_ids.len() {} do not match",
                db.num_divisions,
                db.codebook_ids.len(),
            )));
        }
        let db = Database {
            fs,
            vector_size,
            num_partitions,
            num_divisions,
            num_codes,
            partition_ids: db.partition_ids,
            partitions: RefCell::new(vec![None; num_partitions]),
            partition_centroids_id: db.partition_centroids_id,
            partition_centroids: OnceCell::new(),
            codebook_ids: db.codebook_ids,
            codebooks: RefCell::new(None),
            attributes_log_ids: db.attributes_log_ids,
            attributes_log_load_flags:
                RefCell::new(vec![false; num_partitions]),
            attribute_table: RefCell::new(None),
        };
        Ok(db)
    }
}

/// Partition.
///
/// Bears the centroid element type `T`, but the centroid is not retained
/// because the database manages centroids.
#[derive(Clone)]
pub struct Partition<T> {
    _t: std::marker::PhantomData<T>,
    encoded_vectors: BlockVectorSet<u32>,
    vector_ids: Vec<Uuid>,
}

impl<T> Partition<T> {
    /// Returns the number of vectors in the partition.
    pub fn num_vectors(&self) -> usize {
        self.encoded_vectors.len()
    }

    /// Returns a specified encoded vector.
    ///
    /// `None` if `idnex` ≥ `num_vectors`.
    pub fn get_encoded_vector(&self, index: usize) -> Option<&[u32]> {
        if index < self.encoded_vectors.len() {
            Some(self.encoded_vectors.get(index))
        } else {
            None
        }
    }

    /// Returns the ID of a specified vector.
    ///
    /// `None` if `index` ≥ `num_vectors`.
    pub fn get_vector_id(&self, index: usize) -> Option<&Uuid> {
        self.vector_ids.get(index)
    }
}

/// Interface to load a partition.
///
/// Supposed to be implemented by a specific database.
pub trait LoadPartition<T> {
    /// Loads a partition at a given index.
    ///
    /// `None` if `index` is out of the bounds.
    fn load_partition(&self, index: usize) -> Result<Partition<T>, Error>;
}

/// Codebook.
pub struct Codebook<T> {
    vector_size: usize,
    num_codes: usize,
    codes: Vec<T>, // num_codes × vector_size
}

impl<T> Codebook<T> {
    /// Returns the code vector at a given index.
    ///
    /// `None` if `index` exceeds `num_codes`.
    pub fn get_code_vector(&self, index: usize) -> Option<&[T]> {
        if index < self.num_codes {
            let from = index * self.vector_size;
            let to = from + self.vector_size;
            Some(&self.codes[from..to])
        } else {
            None
        }
    }
}

/// Interface to load a codebook.
///
/// Supposed to be implemented by a specific database.
pub trait LoadCodebook<T> {
    /// Loads a codebook at a given index.
    ///
    /// Fails if `index` is out of the bounds.
    fn load_codebook(&self, index: usize) -> Result<Codebook<T>, Error>;
}

/// Interface to load partition centroids.
///
/// Supposed to be implemented by a specific database.
pub trait LoadPartitionCentroids<T> {
    /// Loads partition centroids.
    ///
    /// Fails if:
    /// - vector size does not match
    /// - number of partitions does not match
    fn load_partition_centroids(&self) -> Result<BlockVectorSet<T>, Error>;
}

/// Attribute table.
pub type AttributeTable = HashMap<Uuid, Attributes>;

/// Events emitted while querying.
pub enum DatabaseQueryEvent {
    StartingQueryInitialization,
    FinishedQueryInitialization,
    StartingPartitionSelection,
    FinishedPartitionSelection,
    StartingPartitionQuery(usize),
    FinishedPartitionQuery(usize),
    StartingResultSelection,
    FinishedResultSelection,
}

/// Query in a specific partition.
struct PartitionQuery<'a, T, FS> {
    database: &'a Database<T, FS>,
    codebooks: Ref<'a, Vec<Codebook<T>>>,
    partition_index: usize,
    localized: Vec<T>, // query vector - partition centroid
}

impl<'a, T, FS> PartitionQuery<'a, T, FS>
where
    T: Scalar,
    FS: FileSystem,
    Database<T, FS>: LoadPartition<T> + LoadCodebook<T>,
{
    fn execute(&self) -> Result<Vec<QueryResult<T>>, Error> {
        let num_divisions = self.database.num_divisions();
        let num_codes = self.database.num_codes();
        let subvector_size = self.database.subvector_size();
        // loads the partition
        let partition = self.database.get_partition(self.partition_index)?;
        // calculates the distance table
        let mut distance_table: Vec<T> =
            Vec::with_capacity(num_divisions * num_codes);
        let mut vector_buf: Vec<T> = Vec::with_capacity(subvector_size);
        unsafe {
            vector_buf.set_len(subvector_size);
        }
        for di in 0..num_divisions {
            let from = di * subvector_size;
            let to = from + subvector_size;
            let subv = &self.localized[from..to];
            let codebook = &self.codebooks[di];
            for ci in 0..num_codes {
                let code_vector = codebook.get_code_vector(ci).unwrap();
                let d = &mut vector_buf[..];
                subtract(subv, code_vector, d);
                distance_table.push(dot(d, d));
            }
        }
        // approximates the squared distances to vectors in the partition
        let num_vectors = partition.num_vectors();
        let mut results: Vec<QueryResult<T>> = Vec::with_capacity(num_vectors);
        for vi in 0..num_vectors {
            let encoded_vector = partition.get_encoded_vector(vi).unwrap();
            let mut distance = T::zero();
            for di in 0..num_divisions {
                let ci = encoded_vector[di] as usize;
                distance += distance_table[di * num_codes + ci];
            }
            results.push(QueryResult {
                partition_index: self.partition_index,
                vector_id: partition.get_vector_id(vi).unwrap().clone(),
                vector_index: vi,
                squared_distance: distance,
                partition_id:
                    self.database.partition_ids[self.partition_index].clone(),
                attributes_log_id:
                    self.database.attributes_log_ids[self.partition_index].clone(),
            });
        }
        Ok(results)
    }
}

/// Query result.
#[derive(Clone, Debug)]
pub struct QueryResult<T> {
    /// Partition index.
    pub partition_index: usize,
    /// Vector ID. Must be unique across the entire database.
    pub vector_id: Uuid,
    /// Vector index. Local index in the partition.
    pub vector_index: usize,
    /// Approximate squared distance.
    pub squared_distance: T,
    // for verification
    partition_id: String,
    attributes_log_id: String,
}
