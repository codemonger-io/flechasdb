//! Vector database.

use core::iter::Iterator;
use core::num::NonZeroUsize;

use crate::error::Error;
use crate::kmeans::{ Codebook, Scalar, cluster };
use crate::linalg::{ dot, subtract_in };
use crate::partitions::{ Partitioning, Partitions };
use crate::slice::AsSlice;
use crate::vector::{ VectorSet, divide_vector_set };

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
        // partitions all the data
        event!(DatabaseBuilderEvent::StartingPartitioning);
        let partitions = self.vs.partition(
            self.num_partitions.try_into().unwrap(),
        )?;
        event!(DatabaseBuilderEvent::FinishedPartitioning);
        // dividing residual vectors
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
            _t: core::marker::PhantomData,
            _vs: core::marker::PhantomData,
            vector_size: partitions.residues.vector_size(),
            num_partitions: self.num_partitions,
            num_divisions: self.num_divisions,
            num_clusters: self.num_clusters,
            partitions,
            codebooks,
        })
    }
}

/// Database builder event.
pub enum DatabaseBuilderEvent {
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
    _t: core::marker::PhantomData<T>,
    _vs: core::marker::PhantomData<VS>,
    // Vector size.
    vector_size: usize,
    // Number of partitions.
    num_partitions: usize,
    // Number of subvector divisions.
    num_divisions: usize,
    // Number of clusters.
    num_clusters: usize,
    // Partitions.
    partitions: Partitions<T, VS>,
    // Codebooks for PQ.
    codebooks: Vec<Codebook<T>>,
}

impl<T, VS> Database<T, VS>
where
    T: Scalar,
    VS: VectorSet<T>,
{
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

    /// Queries k-nearest neighbors (k-NN) of a given vector.
    pub fn query<V, EventHandler>(
        &self, v: &V,
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
        for (i, _) in self.db.partitions.codebook.indices
            .iter()
            .enumerate()
            .filter(|(_, &pi)| pi == self.partition_index)
        {
            let mut distance = T::zero();
            for di in 0..num_divisions {
                let ci = self.db.codebooks[di].indices[i];
                distance += distance_table[di * num_clusters + ci];
            }
            results.push(QueryResult {
                partition_index: self.partition_index,
                vector_index: i,
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
    /// Vector index.
    pub vector_index: usize,
    /// Approximate squared distance.
    pub squared_distance: T,
}
