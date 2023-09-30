//! k-means clustering.

use core::ops::{AddAssign, Div, Mul, MulAssign, Sub, SubAssign};
use core::num::NonZeroUsize;
use rand::Rng;
use rand::distributions::Distribution;
use rand::distributions::uniform::SampleUniform;

use crate::distribution::WeightedIndex;
use crate::error::Error;
use crate::linalg::{add_in, dot, norm2, scale_in, subtract, subtract_in};
use crate::numbers::{Abs, FromAs, Infinity, One, Sqrt, Zero};
use crate::slice::AsSlice;
use crate::vector::{BlockVectorSet, VectorSet};

/// Default epsilon value.
///
/// Specialized for floating point types; i.e., [`f32`], and [`f64`].
pub trait DefaultEpsilon {
    /// Returns the default espsilon value.
    fn default_epsilon() -> Self;
}

impl DefaultEpsilon for f32 {
    fn default_epsilon() -> Self {
        1e-6
    }
}

impl DefaultEpsilon for f64 {
    fn default_epsilon() -> Self {
        1e-12
    }
}

/// Requirements for a vector element as a scalar value.
///
/// [`f32`] and [`f64`] satisfy all of the curated traits.
pub trait Scalar:
    SampleUniform
    + DefaultEpsilon
    + Abs
    + Infinity
    + One
    + Sqrt
    + Zero
    + AddAssign
    + Div<Output = Self>
    + Mul<Output = Self>
    + MulAssign
    + Sub<Output = Self>
    + SubAssign
    + FromAs<usize>
    + PartialOrd
    + Copy
    + core::fmt::Debug {}

impl Scalar for f32 {}
impl Scalar for f64 {}

/// Codebook.
pub struct Codebook<T> {
    /// Cluster centroids.
    pub centroids: BlockVectorSet<T>,

    /// Cluster (centroid) indices assigned to input vectors.
    pub indices: Vec<usize>,
}

/// Event notified while clustering.
#[derive(Debug)]
pub enum ClusterEvent<'a, T> {
    /// Starting centroid initialization.
    StartingCentroidInitialization,
    /// Finished centroid initialization.
    FinishedCentroidInitialization,
    /// Starting n-th centroid update.
    StartingCentroidUpdate(usize),
    /// Finished n-th centroid udpate.
    ///
    /// The second argument is the normalized magnitude of the change in
    /// centroids.
    FinishedCentroidUpdate(usize, &'a T),
    /// Starting n-th centroid reassignment.
    StartingCentroidReassignment(usize),
    /// Finished n-th centroid reassignment.
    FinishedCentroidReassignment(usize),
}

/// Performs k-means clustering.
///
/// Fails if `vs` has fewer vectors than `k`.
pub fn cluster<T, VS>(vs: &VS, k: NonZeroUsize) -> Result<Codebook<T>, Error>
where
    T: Scalar,
    VS: VectorSet<T>,
{
    cluster_with_events(vs, k, |_| {})
}

/// Performs k-means clustering.
///
/// Fails if `vs` has fewer vectors than `k`.
pub fn cluster_with_events<T, VS, EV>(
    vs: &VS,
    k: NonZeroUsize,
    mut event_handler: EV,
) -> Result<Codebook<T>, Error>
where
    T: Scalar,
    VS: VectorSet<T>,
    EV: FnMut(ClusterEvent<'_, T>) -> (),
{
    const R: usize = 100;
    let k = k.get();
    if vs.len() < k {
        return Err(Error::InvalidArgs(
            format!("vs has fewer vectors than k: {} < {}", vs.len(), k),
        ));
    }
    // initializes centroids with k-means++
    event_handler(ClusterEvent::StartingCentroidInitialization);
    let mut codebook = initialize_centroids(vs, k);
    event_handler(ClusterEvent::FinishedCentroidInitialization);
    for r in 0..R {
        // updates centroids
        event_handler(ClusterEvent::StartingCentroidUpdate(r));
        let gradient = update_centroids(vs, &mut codebook);
        event_handler(ClusterEvent::FinishedCentroidUpdate(r, &gradient));
        if gradient < T::default_epsilon() {
            break;
        }
        // re-assigns centroids
        event_handler(ClusterEvent::StartingCentroidReassignment(r));
        reassign_centroids(vs, &mut codebook);
        event_handler(ClusterEvent::FinishedCentroidReassignment(r));
    }
    Ok(codebook)
}

// Initializes centroids and indices with k-means++.
fn initialize_centroids<T, VS>(vs: &VS, k: usize) -> Codebook<T>
where
    T: Scalar,
    VS: VectorSet<T>,
{
    assert!(vs.len() >= k);
    let mut rng = rand::thread_rng();
    let n = vs.len();
    let m = vs.vector_size();
    let mut chosen: Vec<bool> = vec![false; n];
    let mut centroids: Vec<T> = Vec::with_capacity(k * m);
    let mut indices: Vec<usize> = vec![0; n];
    let mut vector_buf: Vec<T> = Vec::with_capacity(m);
    unsafe {
        vector_buf.set_len(m);
    }
    if k == n {
        // no need for clustering
        for i in 0..n {
            centroids.extend_from_slice(vs.get(i).as_slice());
        }
        return Codebook {
            centroids: BlockVectorSet::chunk(
                centroids,
                vs.vector_size().try_into().unwrap(),
            ).unwrap(),
            indices: (0..n).collect(),
        };
    }
    // chooses the first centroid randomly
    let ci = rng.gen_range(0..n);
    chosen[ci] = true;
    let new_centroid = vs.get(ci).as_slice();
    centroids.extend_from_slice(new_centroid);
    if k == 1 {
        return Codebook {
            centroids: BlockVectorSet::chunk(
                centroids,
                vs.vector_size().try_into().unwrap(),
            ).unwrap(),
            indices,
        };
    }
    // calculates the initial distribution
    let mut weights: Vec<T> = Vec::with_capacity(n);
    assert!(n >= 2);
    for i in 0..n {
        if chosen[i] {
            weights.push(T::zero());
        } else {
            let v = vs.get(i).as_slice();
            let d: &mut [T] = &mut vector_buf;
            subtract(v, new_centroid, d);
            let weight = dot(d, d);
            weights.push(weight);
        }
    }
    let mut weighted_index = WeightedIndex::new(weights).unwrap(); // TODO: fails if all the vectors are identical
    // chooses the remaining centroids
    for i in 1..k {
        let ci = weighted_index.sample(&mut rng);
        chosen[ci] = true;
        indices[ci] = i;
        let new_centroid = vs.get(ci).as_slice();
        centroids.extend_from_slice(new_centroid);
        weighted_index.update(&[(ci, &T::zero())]).unwrap();
        for j in 0..n {
            if !chosen[j] {
                let v = vs.get(j).as_slice();
                let d: &mut [T] = &mut vector_buf[..];
                subtract(v, new_centroid, d);
                let new_weight = dot(d, d);
                // updates the weight if it is smaller than the current one
                if new_weight < weighted_index.get_weight(j) {
                    weighted_index.update(&[(j, &new_weight)]).unwrap();
                    indices[j] = i;
                }
            }
        }
    }
    Codebook {
        centroids: BlockVectorSet::chunk(
            centroids,
            vs.vector_size().try_into().unwrap(),
        ).unwrap(),
        indices,
    }
}

// Updates centroids.
fn update_centroids<T, VS>(vs: &VS, codebook: &mut Codebook<T>) -> T
where
    T: Scalar,
    VS: VectorSet<T>,
{
    let m = vs.vector_size();
    let k = codebook.centroids.len();
    let mut vector_buf: Vec<T> = Vec::with_capacity(m);
    unsafe {
        vector_buf.set_len(m);
    }
    let mut max_distance = T::zero();
    let mut max_norm2 = T::zero();
    for i in 0..k {
        let old_centroid = &mut vector_buf[..];
        old_centroid.copy_from_slice(codebook.centroids.get(i));
        let new_centroid = codebook.centroids.get_mut(i);
        new_centroid.fill(T::zero());
        let mut count: usize = 0;
        for (j, _) in codebook.indices
            .iter()
            .enumerate()
            .filter(|(_, &ci)| ci == i)
        {
            add_in(new_centroid, vs.get(j).as_slice());
            count += 1;
        }
        assert_ne!(count, 0);
        scale_in(new_centroid, T::one() / T::from_as(count));
        let centroid_norm2 = norm2(new_centroid);
        if max_norm2 < centroid_norm2 {
            max_norm2 = centroid_norm2
        }
        subtract_in(old_centroid, new_centroid);
        let distance = norm2(old_centroid);
        if max_distance < distance {
            max_distance = distance;
        }
    }
    if max_norm2 != T::zero() {
        max_distance / max_norm2
    } else {
        T::zero()
    }
}

// Re-assigns centroids.
fn reassign_centroids<T, VS>(vs: &VS, codebook: &mut Codebook<T>)
where
    T: Scalar,
    VS: VectorSet<T>,
{
    let n = vs.len();
    let m = vs.vector_size();
    let k = codebook.centroids.len();
    let mut vector_buf: Vec<T> = Vec::with_capacity(m);
    unsafe {
        vector_buf.set_len(m);
    }
    for i in 0..n {
        let v = vs.get(i).as_slice();
        let d = &mut vector_buf[..];
        let mut min_distance = T::infinity();
        let mut min_index: Option<usize> = None;
        for j in 0..k {
            subtract(v, codebook.centroids.get(j).as_slice(), d);
            let distance = dot(d, d);
            if distance < min_distance {
                min_distance = distance;
                min_index = Some(j);
            }
        }
        codebook.indices[i] = min_index.unwrap();
    }
}
