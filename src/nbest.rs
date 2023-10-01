//! N-best selection.

use core::iter::Iterator;
use core::ops::{Deref, DerefMut};

/// N-best elements.
pub struct NBestByKey<T, K, F>
where
    F: FnMut(&T) -> K,
{
    n: usize,
    f: F,
    candidates: Vec<T>,
}

impl<T, K, F> NBestByKey<T, K, F>
where
    F: FnMut(&T) -> K,
{
    /// Creates a new [`NBestByKey`].
    pub fn new(n: usize, f: F) -> Self {
        Self {
            n,
            f,
            candidates: Vec::with_capacity(n),
        }
    }

    /// Consumes the [`NBestByKey`] and returns the underlying vector.
    pub fn into_vec(self) -> Vec<T> {
        self.candidates
    }
}

impl<T, K, F> NBestByKey<T, K, F>
where
    F: FnMut(&T) -> K,
    K: PartialOrd,
{
    /// Pushes a new candidate to the n-best.
    ///
    /// If there are less than `n` candidates, the candidate is pushed to the
    /// current candidates.
    ///
    /// Replaces an item in the current n-best with `candidate`, such that
    /// `candidate` is less than that item (in terms of `F(&T) -> K`).
    /// That replaced item further replaces another item in the current n-best,
    /// such that the replaced item is less than that item.
    /// Repeats the above process until no replacement occurs.
    ///
    /// Returns the item pushed out unless there are less than `n` candidates.
    pub fn push(&mut self, mut candidate: T) -> Option<T> {
        if self.candidates.len() < self.n {
            self.candidates.push(candidate);
            return None;
        }
        while let Some(to_replace) = self.candidates
            .iter_mut()
            .find(|item| (self.f)(&candidate).lt(&(self.f)(item)))
        {
            std::mem::swap(&mut candidate, to_replace);
        }
        Some(candidate)
    }
}

impl<T, K, F> Deref for NBestByKey<T, K, F>
where
    F: FnMut(&T) -> K,
{
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.candidates
    }
}

impl<T, K, F> DerefMut for NBestByKey<T, K, F>
where
    F: FnMut(&T) -> K,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.candidates
    }
}

impl<T, K, F> Into<Vec<T>> for NBestByKey<T, K, F>
where
    F: FnMut(&T) -> K,
{
    fn into(self) -> Vec<T> {
        self.into_vec()
    }
}

impl<T, K, F> IntoIterator for NBestByKey<T, K, F>
where
    F: FnMut(&T) -> K,
{
    type Item = T;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.candidates.into_iter()
    }
}

/// Trait to select n-best elements in a collection.
///
/// Intended to be implemented for [`Iterator`].
pub trait TakeNBestByKey<T>
where
    Self: Sized,
{
    /// Selects n-best elements in the iterator.
    fn n_best_by_key<K, F>(self, n: usize, f: F) -> NBestByKey<T, K, F>
    where
        F: FnMut(&T) -> K,
        K: PartialOrd;
}

impl<I, T> TakeNBestByKey<T> for I
where
    I: Iterator<Item = T> + Sized
{
    fn n_best_by_key<K, F>(self, n: usize, f: F) -> NBestByKey<T, K, F>
    where
        F: FnMut(&T) -> K,
        K: PartialOrd,
    {
        let mut n_best = NBestByKey::new(n, f);
        for item in self {
            n_best.push(item);
        }
        n_best
    }
}
