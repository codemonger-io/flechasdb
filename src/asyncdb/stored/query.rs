//! Query in a stored database.

use core::borrow::Borrow;
use core::future::Future;
use core::hash::Hash;
use core::num::NonZeroUsize;
use core::pin::Pin;
use core::task::{Context, Poll};
use pin_project_lite::pin_project;
use uuid::Uuid;

use crate::error::Error;
use crate::kmeans::Scalar;
use crate::linalg::{dot, subtract};
use crate::slice::AsSlice;
use crate::vector::BlockVectorSet;

use super::{
    Database,
    LoadCodebook,
    LoadPartition,
    LoadPartitionCentroids,
    Partition,
};
use super::get_attribute::GetAttributeInPartition;

pin_project! {
    /// Future that asynchronously runs a query.
    #[must_use = "futures do nothing unless you `.await` or poll them"]
    pub struct Query<'db, 'v, T, FS, V, EV>
    where
        T: Send,
        FS: Send,
        V: Send,
        V: ?Sized,
    {
        db: &'db Database<T, FS>,
        v: &'v V,
        k: usize,
        nprobe: usize,
        event_handler: EV,
        partition_centroids: Option<&'db BlockVectorSet<T>>,
        #[pin]
        load_partition_centroids: Option<Pin<Box<
            dyn 'db + Future<Output = Result<&'db BlockVectorSet<T>, Error>>,
        >>>,
        codebooks: Option<&'db Vec<BlockVectorSet<T>>>,
        #[pin]
        load_codebooks: Option<Pin<Box<
            dyn 'db + Future<Output = Result<&'db Vec<BlockVectorSet<T>>, Error>>,
        >>>,
        partition_queries: Vec<Pin<Box<PartitionQuery<'db, T>>>>,
    }
}

/// Query result.
///
/// Can be derefed as a [`PartitionQueryResult`].
pub struct QueryResult<'db, T, FS>
where
    T: Send,
    FS: Send,
{
    db: &'db Database<T, FS>,
    result: PartitionQueryResult<T>,
}

impl<'db, T, FS> QueryResult<'db, T, FS>
where
    T: Send,
    FS: Send,
{
    fn new(db: &'db Database<T, FS>, result: PartitionQueryResult<T>) -> Self {
        Self {
            db,
            result,
        }
    }
}

impl<'db, T, FS> QueryResult<'db, T, FS>
where
    T: Send,
    FS: Send,
{
    /// Returns an attribue value of the vector corresponding to the result.
    ///
    /// The first call of this function on a result belonging to a partition
    /// will take longer because it will load the attributes of the partition.
    pub fn get_attribute<'i, 'k, K>(
        &'i self,
        key: &'k K,
    ) -> GetAttributeInPartition<'db, 'i, 'k, T, FS, K>
    where
        String: Borrow<K>,
        K: Hash + Eq + Send + ?Sized,
        'i: 'db,
    {
        GetAttributeInPartition::new(
            self.db,
            self.partition_index,
            &self.vector_id,
            key,
        )
    }
}

impl<'db, T, FS> core::ops::Deref for QueryResult<'db, T, FS>
where
    T: Send,
    FS: Send,
{
    type Target = PartitionQueryResult<T>;

    fn deref(&self) -> &Self::Target {
        &self.result
    }
}

// Partition index, localized vector, and squared distance.
struct PartitionVector<T>(usize, Vec<T>, T);

pin_project! {
    // State of a query in a partition.
    struct PartitionQuery<'db, T> {
        vector: PartitionVector<T>,
        #[pin]
        load_partition: Pin<Box<
            dyn 'db + Future<Output = Result<&'db Partition<T>, Error>>,
        >>,
        partition: Option<&'db Partition<T>>,
        results: Option<Vec<PartitionQueryResult<T>>>,
    }
}

/// Result of a query in a single partition.
#[derive(Clone, Debug)]
pub struct PartitionQueryResult<T> {
    /// Index of the partition.
    pub partition_index: usize,
    /// Index of the vector in the partition.
    pub vector_index: usize,
    /// Unique ID of the vector.
    pub vector_id: Uuid,
    /// Approximate squared distance from the query vector.
    pub squared_distance: T,
}

/// Event notified while querying.
pub enum QueryEvent {
    /// Starting to load all the partition centroids.
    StartingLoadingPartitionCentroids,
    /// Finished loading all the partition centroids.
    FinishedLoadingPartitionCentroids,
    /// Starting to load all the codebooks.
    StartingLoadingCodebooks,
    /// Finished loading all the codebooks.
    FinishedLoadingCodebooks,
    /// Starting to select partitions to query.
    StartingPartitionSelection,
    /// Finished selecting partitions to query.
    FinishedPartitionSelection,
    /// Starting to load a single partition at a given index.
    StartingLoadingPartition(usize),
    /// Finished loading a single partition at a given index.
    FinishedLoadingPartition(usize),
    /// Starting to run query on a single partition at a given index.
    StartingPartitionQueryExecution(usize),
    /// Finished running query on a single partition at a given index.
    FinishedPartitionQueryExecution(usize),
    /// Starting to select k-nearest neighbors (k-NN).
    StartingKNNSelection,
    /// Finished selecting k-nearest neighbors (k-NN).
    FinishedKNNSelection,
}

impl<'db, 'v, T, FS, V, EV> Query<'db, 'v, T, FS, V, EV>
where
    T: Send,
    FS: Send,
    V: Send + ?Sized,
{
    /// Creates a new query.
    pub fn new(
        db: &'db Database<T, FS>,
        v: &'v V,
        k: NonZeroUsize,
        nprobe: NonZeroUsize,
        event_handler: EV,
    ) -> Self {
        Query {
            db,
            v,
            k: k.get(),
            nprobe: nprobe.get(),
            event_handler,
            partition_centroids: None,
            load_partition_centroids: None,
            codebooks: None,
            load_codebooks: None,
            partition_queries: Vec::with_capacity(nprobe.get()),
        }
    }
}

impl<'db, 'v, T, FS, V, EV> Future for Query<'db, 'v, T, FS, V, EV>
where
    T: Scalar + Send,
    FS: Send,
    V: AsSlice<T> + Send + ?Sized,
    EV: FnMut(QueryEvent),
    Database<T, FS>:
        LoadPartitionCentroids<'db, T>
        + LoadCodebook<T>
        + LoadPartition<'db, T>,
{
    type Output = Result<Vec<QueryResult<'db, T, FS>>, Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();

        macro_rules! event {
            ($event:expr) => {
                (this.event_handler)($event)
            };
        }

        loop {
            let mut had_progress = false;
            // lazily loads partition centroids and codebooks
            if let Some(partition_centroids) = this.partition_centroids {
                // selects partitions to query and starts loading them
                if this.partition_queries.is_empty() {
                    event!(QueryEvent::StartingPartitionSelection);
                    let selected_partitions = select_partitions(
                        partition_centroids,
                        *this.v,
                        *this.nprobe,
                    );
                    event!(QueryEvent::FinishedPartitionSelection);
                    if selected_partitions.is_empty() {
                        return Poll::Ready(Err(Error::InvalidContext(format!(
                            "no partitions selected for query",
                        ))));
                    }
                    this.partition_queries.extend(
                        selected_partitions.into_iter().map(|p| {
                            event!(QueryEvent::StartingLoadingPartition(p.0));
                            Box::pin(PartitionQuery::start(this.db, p))
                        }),
                    );
                    had_progress = true;
                }
            } else {
                if let Some(future) = this.load_partition_centroids
                    .as_mut()
                    .as_pin_mut()
                {
                    match future.poll(cx) {
                        Poll::Ready(Ok(partition_centroids)) => {
                            event!(QueryEvent::FinishedLoadingPartitionCentroids);
                            *this.partition_centroids =
                                Some(partition_centroids);
                            had_progress = true;
                        },
                        Poll::Pending => {},
                        Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                    };
                } else {
                    event!(QueryEvent::StartingLoadingPartitionCentroids);
                    *this.load_partition_centroids = Some(Box::pin(
                        this.db.load_partition_centroids(),
                    ));
                    had_progress = true;
                }
            }
            // lazily loads codebooks
            if this.codebooks.is_none() {
                if let Some(future) = this.load_codebooks
                    .as_mut().as_pin_mut()
                {
                    match future.poll(cx) {
                        Poll::Ready(Ok(codebooks)) => {
                            event!(QueryEvent::FinishedLoadingCodebooks);
                            *this.codebooks = Some(codebooks);
                            had_progress = true;
                        },
                        Poll::Pending => {},
                        Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                    };
                } else {
                    event!(QueryEvent::StartingLoadingCodebooks);
                    *this.load_codebooks = Some(Box::pin(
                        this.db.load_codebooks(),
                    ));
                    had_progress = true;
                }
            }
            // loads partitions and chooses k-NN
            if !this.partition_queries.is_empty() {
                for query in this.partition_queries.iter_mut() {
                    if query.partition.is_none() {
                        match query.as_mut().poll_loading(cx) {
                            Poll::Ready(Ok(_)) => {
                                event!(QueryEvent::FinishedLoadingPartition(
                                    query.partition_index(),
                                ));
                                had_progress = true;
                            },
                            Poll::Pending => {},
                            Poll::Ready(Err(err)) =>
                                return Poll::Ready(Err(err)),
                        }
                    } else if let Some(codebooks) = this.codebooks {
                        if query.results.is_none() {
                            event!(QueryEvent::StartingPartitionQueryExecution(
                                query.partition_index(),
                            ));
                            if let Err(err) = query
                                .as_mut()
                                .execute(codebooks)
                            {
                                return Poll::Ready(Err(err));
                            }
                            event!(QueryEvent::FinishedPartitionQueryExecution(
                                query.partition_index(),
                            ));
                        }
                    }
                }
                let query_completed = this.partition_queries
                    .iter()
                    .all(|q| q.results.is_some());
                if query_completed {
                    // chooses k-NN
                    event!(QueryEvent::StartingKNNSelection);
                    let results = select_knn(this.partition_queries, *this.k);
                    let results: Vec<_> = results
                        .into_iter()
                        .map(|result| QueryResult::new(
                            *this.db,
                            result.clone(),
                        ))
                        .collect();
                    event!(QueryEvent::FinishedKNNSelection);
                    return Poll::Ready(Ok(results));
                }
            }
            if !had_progress {
                return Poll::Pending;
            }
        }
    }
}

impl<'db, T> PartitionQuery<'db, T>
where
    T: Send,
{
    fn start<FS>(
        db: &'db Database<T, FS>,
        vector: PartitionVector<T>,
    ) -> Self
    where
        FS: Send,
        Database<T, FS>: LoadPartition<'db, T>,
    {
        let index = vector.0;
        Self {
            vector,
            load_partition: db.load_partition(index),
            partition: None,
            results: None,
        }
    }

    const fn partition_index(&self) -> usize {
        self.vector.0
    }

    fn query_vector<'a>(&'a self) -> &'a [T] {
        &self.vector.1
    }

    fn poll_loading(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Result<(), Error>> {
        let mut this = self.project();
        match this.load_partition.as_mut().poll(cx) {
            Poll::Ready(Ok(partition)) => {
                *this.partition = Some(partition);
                Poll::Ready(Ok(()))
            },
            Poll::Pending => Poll::Pending,
            Poll::Ready(Err(err)) => Poll::Ready(Err(err)),
        }
    }
}

impl<'db, T> PartitionQuery<'db, T>
where
    T: Scalar + Send,
{
    // Executes the query in the partition.
    //
    // Updates `results` field.
    //
    // Panics if:
    // - partition is not ready
    fn execute(
        &mut self,
        codebooks: &Vec<BlockVectorSet<T>>,
    ) -> Result<(), Error> {
        let partition = self.partition.expect("partition must be loaded");
        let distance_table = self.calculate_distance_table(codebooks)?;
        let num_vectors = partition.num_vectors();
        let num_divisions = partition.num_divisions();
        let mut results: Vec<PartitionQueryResult<T>> =
            Vec::with_capacity(num_vectors);
        for vi in 0..num_vectors {
            let encoded_vector = partition.get_encoded_vector(vi);
            let mut distance = T::zero();
            for di in 0..num_divisions {
                let ci = encoded_vector[di] as usize;
                distance += distance_table.get(di)[ci];
            }
            results.push(PartitionQueryResult {
                partition_index: self.partition_index(),
                vector_index: vi,
                vector_id: partition.get_vector_id(vi).clone(),
                squared_distance: distance,
            });
        }
        self.results = Some(results);
        Ok(())
    }

    // Calculates the distance table for the partition.
    //
    // Fails if:
    // - `codebooks` is empty
    // - a codebook has no code
    // - vector size is not (# of division) × (subvector size)
    // - numbers of codes in codebooks are not the same
    fn calculate_distance_table(
        &self,
        codebooks: &Vec<BlockVectorSet<T>>,
    ) -> Result<BlockVectorSet<T>, Error> {
        let num_divisions = codebooks.len();
        if num_divisions == 0 {
            return Err(Error::InvalidData(format!("no codebooks")));
        }
        let num_codes = codebooks[0].len();
        if num_codes == 0 {
            return Err(Error::InvalidData(format!("no code in codebook")));
        }
        let subvector_size = codebooks[0].vector_size();
        let query_vector = self.query_vector();
        if query_vector.len() != num_divisions * subvector_size {
            return Err(Error::InvalidData(format!(
                "inconsistent vector size: {} and {}",
                query_vector.len(),
                num_divisions * subvector_size,
            )));
        }
        let mut distance_table: Vec<T> =
            Vec::with_capacity(num_divisions * num_codes);
        let mut vector_buf: Vec<T> = Vec::with_capacity(subvector_size);
        unsafe {
            vector_buf.set_len(subvector_size);
        }
        for di in 0..num_divisions {
            let from = di * subvector_size;
            let to = from + subvector_size;
            let subv = &query_vector[from..to];
            let codebook = &codebooks[di];
            if codebook.len() != num_codes {
                return Err(Error::InvalidData(format!(
                    "inconsistent number of codes: {} and {}",
                    codebook.len(),
                    num_codes,
                )));
            }
            if codebook.vector_size() != subvector_size {
                return Err(Error::InvalidData(format!(
                    "inconsistent subvector size: {} and {}",
                    codebook.vector_size(),
                    subvector_size,
                )));
            }
            for ci in 0..num_codes {
                let code_vector = codebook.get(ci);
                let d = &mut vector_buf[..];
                subtract(subv, code_vector, d);
                distance_table.push(dot(d, d));
            }
        }
        BlockVectorSet::chunk(
            distance_table,
            num_codes.try_into().unwrap(),
        )
    }
}

// Selects `nprobe` partitions nearest to a given vector.
//
// Panics if:
// - nprobe is zero.
// - the vector sizes do not match.
fn select_partitions<T, V>(
    partition_centroids: &BlockVectorSet<T>,
    v: &V,
    nprobe: usize,
) -> Vec<PartitionVector<T>>
where
    T: Scalar,
    V: AsSlice<T> + ?Sized
{
    assert!(nprobe > 0);
    let vector_size = partition_centroids.vector_size();
    let num_partitions = partition_centroids.len();
    let v = v.as_slice();
    assert_eq!(vector_size, v.len());
    let mut partition_vectors: Vec<PartitionVector<T>> =
        Vec::with_capacity(num_partitions);
    for pi in 0..num_partitions {
        let mut localized: Vec<T> = Vec::with_capacity(vector_size);
        unsafe {
            localized.set_len(vector_size);
        }
        let centroid = partition_centroids.get(pi);
        subtract(v, centroid, &mut localized[..]);
        let distance = dot(&localized[..], &localized[..]);
        partition_vectors.push(PartitionVector(pi, localized, distance));
    }
    // chooses `nprobe` nearest vectors
    partition_vectors.sort_by(|l, r| l.2.partial_cmp(&r.2).unwrap());
    partition_vectors.truncate(nprobe);
    partition_vectors
}

// Selects k-nearest neighbors in partition query results.
fn select_knn<'a, 'db, T>(
    queries: &'a Vec<Pin<Box<PartitionQuery<'db, T>>>>,
    k: usize,
) -> Vec<&'a PartitionQueryResult<T>>
where
    T: PartialOrd,
{
    assert!(k > 0);
    type PQR<T> = PartitionQueryResult<T>;
    let compare = |l: &&PQR<T>, r: &&PQR<T>| {
        l.squared_distance.partial_cmp(&r.squared_distance).unwrap()
    };
    let mut results: Vec<_> = queries
        .iter()
        .flat_map(|q| {
            let mut subresults: Vec<_> =
                q.results.as_ref().unwrap().iter().collect();
            subresults.sort_by(compare);
            subresults.truncate(k);
            subresults.into_iter()
        })
        .collect();
    results.sort_by(compare);
    results.truncate(k);
    results
}
