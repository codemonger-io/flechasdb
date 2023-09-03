use std::num::NonZeroUsize;

use crate::error::Error;
use crate::slice::AsSlice;

/// Set of vectors of the same size.
pub trait VectorSet<T> {
    /// Vector type.
    ///
    /// Supposed to be a slice or a Vec.
    type Vector: ?Sized + AsSlice<T>;

    /// Returns the number of vectors in the set.
    fn len(&self) -> usize;

    /// Returns the size of each vector.
    fn vector_size(&self) -> usize;

    /// Returns the i-th vector.
    fn get(&self, i: usize) -> &Self::Vector;
}

/// Vectors in a contiguous array.
#[derive(Debug)]
pub struct BlockVectorSet<T> {
    data: Vec<T>,

    /// Vector size.
    pub vector_size: usize,
}

impl<T> BlockVectorSet<T> {
    /// Tries to chunk a given `Vec`.
    ///
    /// Fails if `data.len` is not a multiple of `vector_size`.
    pub fn chunk(
        data: Vec<T>,
        vector_size: NonZeroUsize,
    ) -> Result<Self, Error> {
        let m = vector_size.get();
        if data.is_empty() || data.len() % m == 0 {
            Ok(Self {
                data,
                vector_size: m,
            })
        } else {
            Err(Error::InvalidArgs(format!(
                "data size ({}) is not a multiple of vector size ({})",
                data.len(),
                vector_size,
            )))
        }
    }
}

impl<T> VectorSet<T> for BlockVectorSet<T> {
    type Vector = [T];

    fn len(&self) -> usize {
        self.data.len() / self.vector_size
    }

    fn vector_size(&self) -> usize {
        self.vector_size
    }

    fn get(&self, i: usize) -> &Self::Vector {
        let from = i * self.vector_size;
        let to = from + self.vector_size;
        &self.data[from..to]
    }
}

/// Divides a given vector set into subvector sets.
///
/// Fails if `vs.vector_size()` is not multiple of `d`.
pub fn divide_vector_set<T, VS>(
    vs: &VS,
    d: NonZeroUsize,
) -> Result<Vec<BlockVectorSet<T>>, Error>
where
    T: Copy,
    VS: VectorSet<T>,
{
    let d = d.get();
    if vs.vector_size() % d != 0 {
        return Err(Error::InvalidArgs(format!(
            "vector size ({}) is not divisible by {}",
            vs.vector_size(),
            d,
        )));
    }
    let n = vs.len();
    let m = vs.vector_size() / d;
    let mut divided: Vec<Vec<T>> = Vec::with_capacity(d);
    for _ in 0..d {
        divided.push(Vec::with_capacity(m * n));
    }
    for i in 0..n {
        let v: &[T] = vs.get(i).as_slice();
        for j in 0..d {
            let from = j * m;
            let to = from + m;
            divided[j].extend_from_slice(&v[from..to])
        }
    }
    let divided = divided
        .drain(..)
        .map(|subdata| BlockVectorSet::chunk(
            subdata,
            m.try_into().unwrap(),
        ).unwrap())
        .collect();
    Ok(divided)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_vector_set_can_chunk_10_elements_into_5_vectors_of_2_elements() {
        let v: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ];
        let vs = BlockVectorSet::chunk(v, 2.try_into().unwrap()).unwrap();
        assert_eq!(vs.vector_size(), 2);
        assert_eq!(vs.len(), 5);
        assert_eq!(vs.get(0), &[1.0, 2.0]);
        assert_eq!(vs.get(1), &[3.0, 4.0]);
        assert_eq!(vs.get(2), &[5.0, 6.0]);
        assert_eq!(vs.get(3), &[7.0, 8.0]);
        assert_eq!(vs.get(4), &[9.0, 10.0]);
    }

    #[test]
    fn block_vector_set_can_chunk_empty_vec() {
        let vs = BlockVectorSet::chunk(
            Vec::<f32>::new(),
            10.try_into().unwrap(),
        ).unwrap();
        assert_eq!(vs.vector_size(), 10);
        assert_eq!(vs.len(), 0);
    }

    #[test]
    fn block_vector_set_cannot_chunk_10_elements_by_vectors_of_3_elements() {
        let v: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        ];
        assert!(BlockVectorSet::chunk(v, 3.try_into().unwrap()).is_err())
    }

    #[test]
    fn divide_vector_set_can_divide_5_vectors_of_6_elements_by_2() {
        let v: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
            19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
        ];
        let vs = BlockVectorSet::chunk(v, 6.try_into().unwrap()).unwrap();
        let divided = divide_vector_set(&vs, 2.try_into().unwrap()).unwrap();
        assert_eq!(divided.len(), 2);
        assert_eq!(divided[0].vector_size(), 3);
        assert_eq!(divided[0].len(), 5);
        assert_eq!(divided[0].get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(divided[0].get(1), &[7.0, 8.0, 9.0]);
        assert_eq!(divided[0].get(2), &[13.0, 14.0, 15.0]);
        assert_eq!(divided[0].get(3), &[19.0, 20.0, 21.0]);
        assert_eq!(divided[0].get(4), &[25.0, 26.0, 27.0]);
        assert_eq!(divided[1].vector_size(), 3);
        assert_eq!(divided[1].len(), 5);
        assert_eq!(divided[1].get(0), &[4.0, 5.0, 6.0]);
        assert_eq!(divided[1].get(1), &[10.0, 11.0, 12.0]);
        assert_eq!(divided[1].get(2), &[16.0, 17.0, 18.0]);
        assert_eq!(divided[1].get(3), &[22.0, 23.0, 24.0]);
        assert_eq!(divided[1].get(4), &[28.0, 29.0, 30.0]);
    }

    #[test]
    fn divice_vector_set_can_divide_empty_vector_set() {
        let vs = BlockVectorSet::chunk(
            Vec::<f32>::new(),
            10.try_into().unwrap(),
        ).unwrap();
        let divided = divide_vector_set(&vs, 2.try_into().unwrap()).unwrap();
        assert_eq!(divided.len(), 2);
        assert_eq!(divided[0].vector_size(), 5);
        assert_eq!(divided[0].len(), 0);
        assert_eq!(divided[1].vector_size(), 5);
        assert_eq!(divided[1].len(), 0);
    }

    #[test]
    fn divide_vector_set_cannot_divide_5_vectors_of_4_elements_by_3() {
        let v: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0,
        ];
        let vs = BlockVectorSet::chunk(v, 4.try_into().unwrap()).unwrap();
        assert!(divide_vector_set(&vs, 3.try_into().unwrap()).is_err());
    }
}
