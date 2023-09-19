//! Database built on memory.

use core::borrow::Borrow;
use core::hash::Hash;
use core::iter::{IntoIterator, Iterator};
use core::num::NonZeroUsize;
use std::collections::HashMap;
use std::collections::hash_map::{Entry as HashMapEntry};
use uuid::Uuid;

use crate::error::Error;
use crate::kmeans::{Codebook, Scalar, cluster};
use crate::linalg::{dot, subtract_in};
use crate::partitions::{Partitioning, Partitions};
use crate::slice::AsSlice;
use crate::vector::{BlockVectorSet, VectorSet, divide_vector_set};

use super::{Attributes, AttributeValue};

pub mod proto;

/// Vector database builder.
pub struct DatabaseBuilder<T, VS>
where
    VS: VectorSet<T>,
{
    _t: core::marker::PhantomData<T>,
    // Input vector set.
    vs: VS,
    // Number of partitions.
    num_partitions: usize,
    // Number of subvector divisions.
    num_divisions: usize,
    // Number of clusters for product quantization (PQ).
    num_clusters: usize,
}

impl<T, VS> DatabaseBuilder<T, VS>
where
    T: Scalar,
    VS: VectorSet<T> + Partitioning<T, VS>,
{
    /// Initializes a builder for a given vector set.
    pub fn new(vs: VS) -> Self {
        Self {
            _t: core::marker::PhantomData,
            vs,
            num_partitions: 10,
            num_divisions: 8,
            num_clusters: 16,
        }
    }

    /// Sets the number of partitions.
    pub fn with_partitions(mut self, num_partitions: NonZeroUsize) -> Self {
        self.num_partitions = num_partitions.get();
        self
    }

    /// Sets the number of subvector divisions.
    pub fn with_divisions(mut self, num_divisions: NonZeroUsize) -> Self {
        self.num_divisions = num_divisions.get();
        self
    }

    /// Sets the number of clusters for product quantization (PQ).
    pub fn with_clusters(mut self, num_clusters: NonZeroUsize) -> Self {
        self.num_clusters = num_clusters.get();
        self
    }

    /// Builds the vector database.
    pub fn build<EventHandler>(
        self,
        mut event_handler: Option<EventHandler>,
    ) -> Result<Database<T, VS>, Error>
    where
        EventHandler: FnMut(DatabaseBuilderEvent) -> (),
    {
        macro_rules! event {
            ($event:expr) => {
                event_handler.iter_mut().for_each(|f| f($event));
            };
        }
        // assigns IDs to vectors
        event!(DatabaseBuilderEvent::StartingIdAssignment);
        let mut vector_ids: Vec<Uuid> = Vec::with_capacity(self.vs.len());
        for _ in 0..self.vs.len() {
            vector_ids.push(Uuid::new_v4());
        }
        event!(DatabaseBuilderEvent::FinishedIdAssignment);
        // partitions all the data
        event!(DatabaseBuilderEvent::StartingPartitioning);
        let partitions = self.vs.partition(
            self.num_partitions.try_into().unwrap(),
        )?;
        event!(DatabaseBuilderEvent::FinishedPartitioning);
        // divides residual vectors
        event!(DatabaseBuilderEvent::StartingSubvectorDivision);
        let divided = divide_vector_set(
            &partitions.residues,
            self.num_divisions.try_into().unwrap(),
        )?;
        event!(DatabaseBuilderEvent::FinishedSubvectorDivision);
        // builds codebooks for residues
        let mut codebooks: Vec<Codebook<T>> = Vec::with_capacity(
            self.num_divisions.try_into().unwrap(),
        );
        for (i, subvs) in divided.iter().enumerate() {
            event!(DatabaseBuilderEvent::StartingQuantization(i));
            codebooks.push(cluster(
                subvs,
                self.num_clusters.try_into().unwrap(),
            )?);
            event!(DatabaseBuilderEvent::FinishedQuantization(i));
        }
        Ok(Database {
            vector_size: partitions.residues.vector_size(),
            num_partitions: self.num_partitions,
            num_divisions: self.num_divisions,
            num_clusters: self.num_clusters,
            vector_ids,
            partitions,
            codebooks,
            attribute_table: HashMap::new(),
        })
    }
}

/// Database builder event.
pub enum DatabaseBuilderEvent {
    StartingIdAssignment,
    FinishedIdAssignment,
    StartingPartitioning,
    FinishedPartitioning,
    StartingSubvectorDivision,
    FinishedSubvectorDivision,
    StartingQuantization(usize),
    FinishedQuantization(usize),
}

/// Database.
pub struct Database<T, VS>
where
    VS: VectorSet<T>,
{
    // Vector size.
    vector_size: usize,
    // Number of partitions.
    num_partitions: usize,
    // Number of subvector divisions.
    num_divisions: usize,
    // Number of clusters.
    num_clusters: usize,
    // Vector IDs.
    vector_ids: Vec<Uuid>,
    // Partitions.
    partitions: Partitions<T, VS>,
    // Codebooks for PQ.
    codebooks: Vec<Codebook<T>>,
    // Attributes associated with vectors.
    attribute_table: HashMap<Uuid, Attributes>,
}

impl<T, VS> Database<T, VS>
where
    VS: VectorSet<T>,
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

    /// Returns the size of a subvector.
    pub fn subvector_size(&self) -> usize {
        self.vector_size / self.num_divisions
    }

    /// Returns the number of clusters.
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Returns an iterator of vector IDs.
    pub fn vector_ids(&self) -> impl Iterator<Item = &Uuid> {
        self.vector_ids.iter()
    }

    /// Returns an iterator of partitions.
    pub fn partitions(&self) -> PartitionIter<'_, T, VS> {
        PartitionIter {
            database: self,
            next_index: 0,
        }
    }

    /// Returns the attribute value of a given vector.
    ///
    /// Fails if no vector is associated with `id`.
    pub fn get_attribute<K>(
        &self,
        id: &Uuid,
        key: &K,
    ) -> Result<Option<&AttributeValue>, Error>
    where
        String: Borrow<K>,
        K: Hash + Eq + ?Sized,
    {
        Ok(
            self.attribute_table
                .get(id)
                .ok_or(Error::InvalidArgs(
                    format!("no such vector ID: {}", id),
                ))?
                .get(key),
            )
    }

    /// Sets an attribute value for the i-th vector.
    ///
    /// Replaces with the new value if the vector already has the attribute.
    ///
    /// Fails if `i` is out of bounds.
    pub fn set_attribute_at<KV, KEY, VAL>(
        &mut self,
        i: usize,
        attribute: KV,
    ) -> Result<(), Error>
    where
        KV: Into<(KEY, VAL)>,
        KEY: Into<String>,
        VAL: Into<AttributeValue>,
    {
        let id = self.vector_ids.get(i)
            .ok_or(Error::InvalidArgs(
                format!("vector index out of bounds: {}", i),
            ))?;
        let (key, value) = attribute.into();
        let key = key.into();
        let value = value.into();
        if let Some(attributes) = self.attribute_table.get_mut(id) {
            match attributes.entry(key.into()) {
                HashMapEntry::Occupied(entry) => {
                    *entry.into_mut() = value.into();
                },
                HashMapEntry::Vacant(entry) => {
                    entry.insert(value.into());
                },
            };
        } else {
            self.attribute_table.insert(
                id.clone(),
                Attributes::from([(key, value)]),
            );
        }
        Ok(())
    }
}

impl<T, VS> Database<T, VS>
where
    T: Scalar,
    VS: VectorSet<T>,
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
        V: AsSlice<T> + ?Sized,
        EventHandler: FnMut(DatabaseQueryEvent) -> (),
    {
        macro_rules! event {
            ($event:expr) => {
                event_handler.iter_mut().for_each(|f| f($event));
            };
        }
        event!(DatabaseQueryEvent::StartingPartitionSelection);
        let v = v.as_slice();
        let queries = self.query_partitions(v, nprobe)?;
        event!(DatabaseQueryEvent::FinishedPartitionSelection);
        let mut all_results: Vec<QueryResult<T>> = Vec::new();
        for query in &queries {
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

    // Queries partitions.
    //
    // Fails if `nprobe` exceeds the number of partitions.
    fn query_partitions<'a>(
        &'a self,
        v: &[T],
        nprobe: NonZeroUsize,
    ) -> Result<Vec<PartitionQuery<'a, T, VS>>, Error> {
        let nprobe = nprobe.get();
        if nprobe > self.num_partitions {
            return Err(Error::InvalidArgs(format!(
                "nprobe {} exceeds the number of partitions {}",
                nprobe,
                self.num_partitions,
            )));
        }
        // localizes vectors and calculates distances
        let mut local_vectors: Vec<(usize, Vec<T>, T)> =
            Vec::with_capacity(self.num_partitions);
        for pi in 0..self.num_partitions {
            let mut localized: Vec<T> = Vec::new();
            localized.extend_from_slice(v);
            let centroid = self.partitions.codebook.centroids.get(pi);
            subtract_in(&mut localized[..], centroid.as_slice());
            let distance = dot(&localized[..], &localized[..]);
            local_vectors.push((pi, localized, distance));
        }
        // chooses `nprobe` shortest distances
        local_vectors.sort_by(|lhs, rhs| lhs.2.partial_cmp(&rhs.2).unwrap());
        local_vectors.truncate(nprobe);
        // queries
        let queries = local_vectors
            .into_iter()
            .map(|(partition_index, localized, _)| PartitionQuery {
                db: self,
                partition_index,
                localized,
            })
            .collect();
        Ok(queries)
    }
}

/// Iterator of partitions in a database.
pub struct PartitionIter<'a, T, VS>
where
    VS: VectorSet<T>,
{
    // Database.
    database: &'a Database<T, VS>,
    // Next partition index.
    next_index: usize,
}

impl<'a, T, VS> Iterator for PartitionIter<'a, T, VS>
where
    T: Clone,
    VS: VectorSet<T>,
{
    type Item = Partition<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index < self.database.num_partitions {
            let partition = Partition::new(self.database, self.next_index);
            self.next_index += 1;
            Some(partition)
        } else {
            None
        }
    }
}

/// Partition in a database.
pub struct Partition<T> {
    // Centroid of the partition.
    pub centroid: Vec<T>,
    // Encoded vectors.
    pub encoded_vectors: BlockVectorSet<u32>,
    // Vector IDs.
    pub vector_ids: Vec<Uuid>,
}

impl<T> Partition<T> {
    /// Returns the vector size.
    pub fn vector_size(&self) -> usize {
        self.centroid.len()
    }

    /// Returns the number of subvector divisions.
    pub fn num_divisions(&self) -> usize {
        self.encoded_vectors.vector_size()
    }

    /// Returns the number of vectors.
    pub fn num_vectors(&self) -> usize {
        self.encoded_vectors.len()
    }
}

impl<T> Partition<T>
where
    T: Clone,
{
    /// Extracts a partition from a given database.
    fn new<VS>(db: &Database<T, VS>, index: usize) -> Self
    where
        VS: VectorSet<T>,
    {
        let mut centroid: Vec<T> = Vec::with_capacity(db.vector_size());
        centroid.extend_from_slice(
            db.partitions.codebook.centroids.get(index),
        );
        let num_divisions = db.num_divisions();
        let num_vectors = db.partitions.codebook.indices
            .iter()
            .filter(|&&pi| pi == index)
            .count();
        let mut encoded_vectors: Vec<u32> =
            Vec::with_capacity(num_vectors * num_divisions);
        let mut vector_ids: Vec<Uuid> = Vec::with_capacity(num_vectors);
        for (vi, _) in db.partitions.codebook.indices
            .iter()
            .enumerate()
            .filter(|(_, &pi)| pi == index)
        {
            for di in 0..num_divisions {
                encoded_vectors.push(
                    db.codebooks[di].indices[vi].try_into().unwrap(),
                );
            }
            vector_ids.push(db.vector_ids[vi]);
        }
        Partition {
            centroid,
            encoded_vectors: BlockVectorSet::chunk(
                encoded_vectors,
                num_divisions.try_into().unwrap(),
            ).unwrap(),
            vector_ids,
        }
    }
}

/// Database query event.
pub enum DatabaseQueryEvent {
    StartingPartitionSelection,
    FinishedPartitionSelection,
    StartingPartitionQuery(usize),
    FinishedPartitionQuery(usize),
    StartingResultSelection,
    FinishedResultSelection,
}

/// Query in a partition.
pub struct PartitionQuery<'a, T, VS>
where
    VS: VectorSet<T>,
{
    // Database.
    db: &'a Database<T, VS>,
    // Partition index.
    partition_index: usize,
    // Localized query vector.
    localized: Vec<T>,
}

impl<'a, T, VS> PartitionQuery<'a, T, VS>
where
    T: Scalar,
    VS: VectorSet<T>,
{
    /// Executes the query.
    pub fn execute(&self) -> Result<Vec<QueryResult<T>>, Error> {
        let num_divisions = self.db.num_divisions();
        let num_clusters = self.db.num_clusters();
        let md = self.db.subvector_size();
        // calculates the distance table
        let mut distance_table: Vec<T> = Vec::with_capacity(
            num_divisions * num_clusters,
        );
        let mut vector_buf = vec![T::zero(); md];
        for di in 0..num_divisions {
            let from = di * md;
            let to = from + md;
            let subv = &self.localized[from..to];
            for ci in 0..num_clusters {
                let centroid = self.db.codebooks[di].centroids.get(ci);
                let d = &mut vector_buf[..];
                d.copy_from_slice(subv);
                subtract_in(d, centroid.as_slice());
                distance_table.push(dot(d, d));
            }
        }
        // approximates the squared distances to individual vectors
        let mut results: Vec<QueryResult<T>> = Vec::with_capacity(
            self.partition_size(),
        );
        for (pvi, (vi, _)) in self.db.partitions.codebook.indices
            .iter()
            .enumerate()
            .filter(|(_, &pi)| pi == self.partition_index)
            .enumerate()
        {
            let mut distance = T::zero();
            for di in 0..num_divisions {
                let ci = self.db.codebooks[di].indices[vi];
                distance += distance_table[di * num_clusters + ci];
            }
            results.push(QueryResult {
                partition_index: self.partition_index,
                vector_id: self.db.vector_ids[vi].clone(),
                vector_index: pvi,
                squared_distance: distance,
            });
        }
        Ok(results)
    }

    /// Returns the partition size.
    fn partition_size(&self) -> usize {
        self.db.partitions.codebook.indices
            .iter()
            .filter(|pi| **pi == self.partition_index)
            .count()
    }
}

/// Query result.
#[derive(Clone, Debug)]
pub struct QueryResult<T> {
    /// Partition index.
    pub partition_index: usize,
    /// Vector ID. Must be unique across the database.
    pub vector_id: Uuid,
    /// Vector index. Local index in the partition.
    pub vector_index: usize,
    /// Approximate squared distance.
    pub squared_distance: T,
}
