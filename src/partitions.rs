//! Partitions.
//!
//! Reference:
//! - <https://mccormickml.com/2017/10/22/product-quantizer-tutorial-part-2/>

use core::num::NonZeroUsize;

use crate::error::Error;
use crate::kmeans::{ClusterEvent, Codebook, Scalar, cluster_with_events};
use crate::linalg::{add_in, subtract_in};
use crate::slice::AsSlice;
use crate::vector::{BlockVectorSet, VectorSet};

/// Partitions.
///
/// Moves the input vector set to save memory.
pub struct Partitions<T, VS> {
    /// Codebook of the partition.
    pub codebook: Codebook<T>,
    /// Residue vector set.
    pub residues: VS,
}

impl<T, VS> Partitions<T, VS>
where
    T: Scalar,
    VS: VectorSet<T>,
{
    /// Returns the number of partitions.
    pub fn num_partitions(&self) -> usize {
        self.codebook.indices.len()
    }

    /// Returns an iterator of all the input vectors.
    ///
    /// Iteration may be expensive because it reconstructs an input vector
    /// every `next` call.
    pub fn all_vectors(&self) -> AllVectorIterator<'_, T, VS> {
        AllVectorIterator::new(self)
    }
}

/// Iterator of all the input vectors in partitions.
pub struct AllVectorIterator<'a, T, VS>
where
    VS: VectorSet<T>,
{
    // Parent partitions.
    partitions: &'a Partitions<T, VS>,
    // Next index.
    next_index: usize,
}

impl<'a, T, VS> AllVectorIterator<'a, T, VS>
where
    T: Scalar,
    VS: VectorSet<T>,
{
    /// Starts iteration of all vectors in given partitions.
    pub fn new(partitions: &'a Partitions<T, VS>) -> Self {
        Self {
            partitions,
            next_index: 0,
        }
    }
}

impl<'a, T, VS> Iterator for AllVectorIterator<'a, T, VS>
where
    T: Scalar,
    VS: VectorSet<T>,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index < self.partitions.residues.len() {
            let mut v: Vec<T> = Vec::with_capacity(
                self.partitions.residues.vector_size(),
            );
            v.extend_from_slice(
                self.partitions.residues.get(self.next_index).as_slice(),
            );
            let codebook = &self.partitions.codebook;
            let ci = codebook.indices[self.next_index];
            let centroid = codebook.centroids.get(ci).as_slice();
            add_in(&mut v[..], centroid);
            self.next_index += 1;
            Some(v)
        } else {
            None
        }
    }
}

/// Implementation of partitioning in place.
pub trait Partitioning<T, VS>
where
    Self: Sized,
{
    /// Partitions the vector set in place.
    fn partition(self, p: NonZeroUsize) -> Result<Partitions<T, VS>, Error> {
        self.partition_with_events(p, |_| ())
    }

    /// Partitions the vector set in place.
    fn partition_with_events<EV>(
        self,
        p: NonZeroUsize,
        event_handler: EV,
    ) -> Result<Partitions<T, VS>, Error>
    where
        EV: FnMut(ClusterEvent<'_, T>) -> ();
}

impl<T> Partitioning<T, Self> for BlockVectorSet<T>
where
    T: Scalar,
{
    fn partition_with_events<EV>(
        mut self,
        p: NonZeroUsize,
        event_handler: EV,
    ) -> Result<Partitions<T, Self>, Error>
    where
        EV: FnMut(ClusterEvent<'_, T>) -> (),
    {
        let codebook = cluster_with_events(&self, p, event_handler)?;
        for i in 0..p.get() {
            let centroid = codebook.centroids.get(i);
            for (j, _) in codebook.indices
                .iter()
                .enumerate()
                .filter(|(_, &ci)| ci == i)
            {
                let v = self.get_mut(j);
                subtract_in(v, centroid);
            }
        }
        Ok(Partitions {
            codebook,
            residues: self,
        })
    }
}
