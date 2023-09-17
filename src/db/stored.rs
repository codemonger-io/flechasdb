//! Defines a stored database.

use core::borrow::Borrow;
use core::cell::{Ref, RefCell};
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
};
use crate::protos::{Deserialize, read_message};
use crate::slice::AsSlice;

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
    partition_centroids: Vec<T>, // num_partitions × vector_size.
    codebook_ids: Vec<String>,
    codebooks: RefCell<Option<Vec<Codebook<T>>>>,
    attributes_log_ref: String,
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

    /// Returns the centroid of a partition.
    ///
    /// `None` if `index` ≥ `num_partitions`.
    pub fn get_partition_centroid(&self, index: usize) -> Option<&[T]> {
        if index < self.num_partitions {
            let from = index * self.vector_size;
            let to = from + self.vector_size;
            Some(&self.partition_centroids[from..to])
        } else {
            None
        }
    }

    /// Returns a code vector in a specified division.
    ///
    /// `None` if `index` ≥ `num_divisons`.
    pub fn get_codebook_id(&self, index: usize) -> Option<&String> {
        self.codebook_ids.get(index)
    }

    /// Returns an attribute value of a given vector.
    ///
    /// Fails if no vector is associated with `id`.
    ///
    /// `None` if the vector exists but no value is associated with `key`.
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
            let time = std::time::Instant::now();
            self.load_attribute_table()?;
            println!("loaded attribute table in {} μs", time.elapsed().as_micros());
        }
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
        let mut path = PathBuf::from("attributes");
        path.push(&self.attributes_log_ref);
        path.set_extension(PROTOBUF_EXTENSION);
        let mut f = self.fs.open_hashed_file(path)?;
        let attributes_log: ProtosAttributesLog = read_message(&mut f)?;
        let mut attribute_table: AttributeTable = AttributeTable::new();
        for (i, entry) in attributes_log.entries.into_iter().enumerate() {
            let vector_id = entry.vector_id
                .into_option()
                .ok_or(Error::InvalidData(
                    format!("attributes log[{}]: missing vector ID", i),
                ))?
                .deserialize()?;
            let value = entry.value
                .into_option()
                .ok_or(Error::InvalidData(
                    format!("attributes log[{}]: missing value", i),
                ))?
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
        self.attribute_table.replace(Some(attribute_table));
        Ok(())
    }
}

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
        let num_vectors = partition.num_vectors as usize;
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
        if num_vectors != partition.encoded_vectors.len() {
            return Err(Error::InvalidData(format!(
                "number of encoded vectors is inconsistent: expected {} but got {}",
                num_vectors,
                partition.encoded_vectors.len(),
            )));
        }
        if num_vectors != partition.vector_ids.len() {
            return Err(Error::InvalidData(format!(
                "number of vector IDs is inconsistent: exptected {} but got {}",
                num_vectors,
                partition.vector_ids.len(),
            )));
        }
        // lodas encoded vectors.
        let mut encoded_vectors: Vec<u32> =
            Vec::with_capacity(num_vectors * num_divisions);
        for encoded_vector in partition.encoded_vectors.into_iter() {
            if encoded_vector.elements.len() != num_divisions {
                return Err(Error::InvalidData(format!(
                    "num_divisions {} and encoded vector length {} do not match",
                    num_divisions,
                    encoded_vector.elements.len(),
                )));
            }
            encoded_vectors.extend(encoded_vector.elements);
        }
        // loads vector IDs 
        let vector_ids: Vec<Uuid> = partition.vector_ids
            .into_iter()
            .map(|id| id.deserialize().unwrap())
            .collect();
        Ok(Partition {
            _t: std::marker::PhantomData,
            num_divisions,
            num_vectors,
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

impl<T, FS> Database<T, FS>
where
    T: Scalar,
    FS: FileSystem,
    Self: LoadPartition<T> + LoadCodebook<T>,
{
    /// Queries k-nearest neighbors (k-NN) of a given vector.
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
        if self.codebooks.borrow().is_none() {
            // loads codebooks if not loaded yet.
            let mut codebooks: Vec<Codebook<T>> =
                Vec::with_capacity(self.num_divisions());
            for di in 0..self.num_divisions() {
                codebooks.push(self.load_codebook(di)?);
            }
            self.codebooks.replace(Some(codebooks));
        }
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
        // localizes vectors and calculates distances
        let mut distances: Vec<(usize, Vec<T>, T)> =
            Vec::with_capacity(num_partitions);
        for pi in 0..num_partitions {
            let mut localized: Vec<T> = Vec::with_capacity(self.vector_size());
            unsafe {
                localized.set_len(self.vector_size());
            }
            let centroid = self.get_partition_centroid(pi).unwrap();
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
        if num_partitions != db.partition_refs.len() {
            return Err(Error::InvalidData(format!(
                "num_partitions {} and partition_refs.len() {} do not match",
                db.num_partitions,
                db.partition_refs.len(),
            )));
        }
        if num_divisions != db.codebook_refs.len() {
            return Err(Error::InvalidData(format!(
                "num_divisions {} and codebook_refs.len() {} do not match",
                db.num_divisions,
                db.codebook_refs.len(),
            )));
        }
        // loads partition IDs and centroids
        let mut partition_ids: Vec<String> = Vec::with_capacity(num_partitions);
        let mut partition_centroids: Vec<f32> =
            Vec::with_capacity(num_partitions * vector_size);
        for partition_ref in db.partition_refs.into_iter() {
            partition_ids.push(partition_ref.id);
            let centroid = partition_ref.centroid;
            if centroid.len() != vector_size {
                return Err(Error::InvalidData(format!(
                    "vector_size {} and centroid size {} do not match",
                    db.vector_size,
                    centroid.len(),
                )));
            }
            partition_centroids.extend(centroid);
        }
        // loads codebook IDs
        let mut codebook_ids: Vec<String> = Vec::with_capacity(num_divisions);
        for codebook_ref in db.codebook_refs.into_iter() {
            codebook_ids.push(codebook_ref.id);
        }
        let db = Database {
            fs,
            vector_size,
            num_partitions,
            num_divisions,
            num_codes,
            partition_ids,
            partition_centroids,
            codebook_ids,
            codebooks: RefCell::new(None),
            attributes_log_ref: db.attributes_log_ref,
            attribute_table: RefCell::new(None),
        };
        Ok(db)
    }
}

/// Partition.
///
/// Bears the centroid element type `T`, but the centroid is not retained
/// because the database manages centroids.
pub struct Partition<T> {
    _t: std::marker::PhantomData<T>,
    num_divisions: usize,
    num_vectors: usize,
    encoded_vectors: Vec<u32>, // num_vectors × num_divisions
    vector_ids: Vec<Uuid>,
}

impl<T> Partition<T> {
    /// Returns the number of vectors in the partition.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Returns a specified encoded vector.
    ///
    /// `None` if `idnex` ≥ `num_vectors`.
    pub fn get_encoded_vector(&self, index: usize) -> Option<&[u32]> {
        if index < self.num_vectors {
            let from = index * self.num_divisions;
            let to = from + self.num_divisions;
            Some(&self.encoded_vectors[from..to])
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
    /// `None` if `index` is out of the bounds.
    fn load_codebook(&self, index: usize) -> Result<Codebook<T>, Error>;
}

/// Attribute table.
pub type AttributeTable = HashMap<Uuid, Attributes>;

/// Events emitted while querying.
pub enum DatabaseQueryEvent {
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
        let partition = self.database.load_partition(self.partition_index)?;
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
}
