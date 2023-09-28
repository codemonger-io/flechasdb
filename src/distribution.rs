//! More straightforward weighted distribution.

use core::cmp::PartialOrd;
use core::ops::{AddAssign, SubAssign};
use rand::Rng;
use rand::distributions::Distribution;
use rand::distributions::uniform::{SampleUniform, UniformSampler};

use crate::error::Error;
use crate::linalg::{min, sum};
use crate::numbers::Zero;

/// Straightforward weighted distribution.
///
/// Unlike [`rand::distributions::weighted::WeightedIndex`], this distribution
/// won't precalculate the cumulative weights, but will do it on the fly.
pub struct WeightedIndex<X>
where
    X: SampleUniform + PartialOrd,
{
    weights: Vec<X>,
    total_weight: X,
    weight_distribution: X::Sampler,
}

impl<X> WeightedIndex<X>
where
    X: SampleUniform + Zero + AddAssign + SubAssign + PartialOrd + Copy,
{
    /// Creates a new [`WeightedIndex`] from a given [`Vec`].
    ///
    /// Fails if `weights` is empty,
    /// or if any of the weights is negative,
    /// or if the sum of all weights is zero.
    pub fn new(weights: Vec<X>) -> Result<Self, Error> {
        if weights.is_empty() {
            return Err(Error::InvalidArgs(format!("weights is empty")));
        }
        let mn = min(&weights[..]).unwrap();
        if mn < X::zero() {
            return Err(Error::InvalidArgs(
                format!("weights contains negagive"),
            ));
        }
        let total_weight = sum(&weights[..]);
        if total_weight <= X::zero() {
            return Err(Error::InvalidArgs(format!("total weight is zero")));
        }
        Ok(Self {
            weights,
            total_weight,
            weight_distribution: X::Sampler::new(X::zero(), total_weight),
        })
    }

    /// Updates the weights.
    ///
    /// Fails if the total weight will become zero,
    /// or if any of the new weight is negative,
    /// or if any of the weight indices is out of range.
    ///
    /// In case update fails, the weights are not changed.
    pub fn update(&mut self, new_weights: &[(usize, &X)]) -> Result<(), Error> {
        // calculates and verifies the total weight before updating weights
        let mut new_total_weight = self.total_weight;
        for &(i, &weight) in new_weights {
            if i >= self.weights.len() {
                return Err(Error::InvalidArgs(format!("index out of range")));
            }
            if weight < X::zero() {
                return Err(Error::InvalidArgs(
                    format!("new weights contains negative"),
                ));
            }
            new_total_weight -= self.weights[i];
            new_total_weight += weight;
        }
        if new_total_weight <= X::zero() {
            return Err(Error::InvalidArgs(
                format!("total weight becomes zero"),
            ));
        }
        // safely updates the weights
        new_weights.iter().for_each(|(i, weight)| self.weights[*i] = **weight);
        self.total_weight = new_total_weight;
        self.weight_distribution = X::Sampler::new(
            X::zero(),
            self.total_weight,
        );
        Ok(())
    }

    /// Returns the weight.
    pub fn get_weight(&self, index: usize) -> X {
        self.weights[index]
    }
}

impl<X> Distribution<usize> for WeightedIndex<X>
where
    X: SampleUniform + Zero + AddAssign + PartialOrd + Copy,
{
    /// Samples a value from the distribution.
    fn sample<R>(&self, rng: &mut R) -> usize
    where
        R: Rng + ?Sized,
    {
        let sample = self.weight_distribution.sample(rng);
        let mut cum_weight = X::zero();
        let mut last_non_zero_index: Option<usize> = None;
        for i in 0..self.weights.len() {
            if self.weights[i] > X::zero() {
                last_non_zero_index = Some(i);
                cum_weight += self.weights[i];
                if cum_weight > sample {
                    break;
                }
            }
        }
        last_non_zero_index.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;
    use core::cmp::Ordering;
    use rand::distributions::uniform::SampleBorrow;

    use super::*;

    // defines an own number type to fake sampler
    #[derive(Clone, Copy, PartialEq)]
    struct Number(f32);
    impl From<f32> for Number {
        fn from(x: f32) -> Self {
            Self(x)
        }
    }
    impl SampleUniform for Number {
        type Sampler = NumberSampler;
    }
    impl Zero for Number {
        fn zero() -> Number {
            Number(0.0)
        }
    }
    impl AddAssign<Number> for Number {
        fn add_assign(&mut self, rhs: Number) {
            self.0 += rhs.0;
        }
    }
    impl SubAssign<Number> for Number {
        fn sub_assign(&mut self, rhs: Number) {
            self.0 -= rhs.0;
        }
    }
    impl PartialOrd for Number {
        fn partial_cmp(&self, other: &Number) -> Option<Ordering> {
            self.0.partial_cmp(&other.0)
        }
    }

    // fake sampler that produces sequential numbers incremented by 0.5
    // in a given range.
    struct NumberSampler {
        low: Number,
        high: Number,
        next: Cell<Number>,
    }
    impl UniformSampler for NumberSampler {
        type X = Number;
        fn new<B1, B2>(low: B1, high: B2) -> Self
        where
            B1: SampleBorrow<Self::X> + Sized,
            B2: SampleBorrow<Self::X> + Sized,
        {
            let low = *low.borrow();
            Self {
                low,
                high: *high.borrow(),
                next: Cell::new(low),
            }
        }
        fn new_inclusive<B1, B2>(_: B1, _: B2) -> Self
        where
            B1: SampleBorrow<Self::X> + Sized,
            B2: SampleBorrow<Self::X> + Sized,
        {
            panic!("should not be used in the test");
        }
        fn sample<R>(&self, rng: &mut R) -> Self::X
        where
            R: Rng + ?Sized,
        {
            let _ = rng.gen::<f32>();
            let next = self.next.get();
            let new_next = Number(next.0 + 0.5);
            if new_next < self.high {
                self.next.set(new_next);
            } else {
                self.next.set(self.low);
            }
            next
        }
    }

    #[test]
    fn weighted_index_should_distribute_indices_according_to_weights() {
        let weights = vec![Number(1.0), Number(3.0), Number(6.0)];
        let weighted_index = WeightedIndex::new(weights).unwrap();
        let rng = rand::thread_rng();
        let indices: Vec<usize> = rng
            .sample_iter(&weighted_index)
            .take(20)
            .collect();
        assert_eq!(indices, &[
            0, 0,
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        ]);
    }

    #[test]
    fn weighted_index_should_not_produce_indices_of_zero_weight() {
        let mut rng = rand::thread_rng();
        let weights = vec![Number(0.0), Number(1.0), Number(2.0)];
        let weighted_index = WeightedIndex::new(weights).unwrap();
        let indices: Vec<usize> = (&mut rng)
            .sample_iter(&weighted_index)
            .take(6)
            .collect();
        assert_eq!(indices, &[1, 1, 2, 2, 2, 2]);
        let weights = vec![Number(1.0), Number(0.0), Number(2.0)];
        let weighted_index = WeightedIndex::new(weights).unwrap();
        let indices: Vec<usize> = (&mut rng)
            .sample_iter(&weighted_index)
            .take(6)
            .collect();
        assert_eq!(indices, &[0, 0, 2, 2, 2, 2]);
        let weights = vec![Number(1.0), Number(2.0), Number(0.0)];
        let weighted_index = WeightedIndex::new(weights).unwrap();
        let indices: Vec<usize> = (&mut rng)
            .sample_iter(&weighted_index)
            .take(6)
            .collect();
        assert_eq!(indices, &[0, 0, 1, 1, 1, 1]);
    }

    #[test]
    fn weighted_index_new_should_fail_if_weights_is_empty() {
        let weights: Vec<f32> = vec![];
        assert!(WeightedIndex::new(weights).is_err());
    }

    #[test]
    fn weighted_index_new_should_fail_if_weights_contains_negative() {
        let weights: Vec<f32> = vec![0.0, -1.0, 2.0];
        assert!(WeightedIndex::new(weights).is_err());
    }

    #[test]
    fn weighted_index_new_should_fail_if_total_weight_is_zero() {
        let weights: Vec<f32> = vec![0.0; 3];
        assert!(WeightedIndex::new(weights).is_err());
    }

    #[test]
    fn weighted_index_should_return_individual_weight() {
        let weights: Vec<f32> = vec![1.0, 3.0, 6.0];
        let weighted_index = WeightedIndex::new(weights).unwrap();
        assert_eq!(weighted_index.get_weight(0), 1.0);
        assert_eq!(weighted_index.get_weight(1), 3.0);
        assert_eq!(weighted_index.get_weight(2), 6.0);
    }

    #[test]
    fn weighted_index_can_update_a_weight() {
        let weights: Vec<f32> = vec![1.0, 3.0, 6.0];
        let mut weighted_index = WeightedIndex::new(weights).unwrap();
        weighted_index.update(&[(0, &2.0)]).unwrap();
        assert_eq!(weighted_index.get_weight(0), 2.0);
        weighted_index.update(&[(1, &1.0)]).unwrap();
        assert_eq!(weighted_index.get_weight(1), 1.0);
        weighted_index.update(&[(2, &0.0)]).unwrap();
        assert_eq!(weighted_index.get_weight(2), 0.0);
    }

    #[test]
    fn weighted_index_can_update_multiple_weights() {
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut weighted_index = WeightedIndex::new(weights).unwrap();
        weighted_index.update(&[(1, &0.0), (2, &1.0), (4, &10.0)]).unwrap();
        assert_eq!(weighted_index.get_weight(0), 1.0);
        assert_eq!(weighted_index.get_weight(1), 0.0);
        assert_eq!(weighted_index.get_weight(2), 1.0);
        assert_eq!(weighted_index.get_weight(3), 4.0);
        assert_eq!(weighted_index.get_weight(4), 10.0);
    }

    #[test]
    fn weighted_index_cannot_update_to_make_total_weight_zero() {
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut weighted_index = WeightedIndex::new(weights).unwrap();
        assert!(
            weighted_index
                .update(&[(0, &0.0), (1, &0.0), (2, &0.0)])
                .is_err(),
        );
        assert_eq!(weighted_index.get_weight(0), 1.0);
        assert_eq!(weighted_index.get_weight(1), 2.0);
        assert_eq!(weighted_index.get_weight(2), 3.0);
    }

    #[test]
    fn weighted_index_cannot_update_to_contain_negative() {
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut weighted_index = WeightedIndex::new(weights).unwrap();
        assert!(weighted_index.update(&[(0, &-1.0)]).is_err());
        assert!(weighted_index.update(&[(1, &-2.0)]).is_err());
        assert!(weighted_index.update(&[(2, &-3.0)]).is_err());
        assert_eq!(weighted_index.get_weight(0), 1.0);
        assert_eq!(weighted_index.get_weight(1), 2.0);
        assert_eq!(weighted_index.get_weight(2), 3.0);
    }

    #[test]
    fn weighted_index_cannot_update_out_of_bounds_weight() {
        let weights: Vec<f32> = vec![1.0, 2.0, 3.0];
        let mut weighted_index = WeightedIndex::new(weights).unwrap();
        assert!(weighted_index.update(&[(3, &1.0)]).is_err());
        assert_eq!(weighted_index.get_weight(0), 1.0);
        assert_eq!(weighted_index.get_weight(1), 2.0);
        assert_eq!(weighted_index.get_weight(2), 3.0);
    }

    #[test]
    fn weighted_index_should_distribute_indices_according_to_updated_weights() {
        let weights = vec![Number(1.0), Number(2.0), Number(3.0)];
        let mut weighted_index = WeightedIndex::new(weights).unwrap();
        weighted_index.update(&[(0, &Number(5.0)), (2, &Number(0.0))]).unwrap();
        let rng = rand::thread_rng();
        let indices = rng
            .sample_iter(&weighted_index)
            .take(14)
            .collect::<Vec<usize>>();
        assert_eq!(indices, &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1,
        ]);
    }

    #[test]
    fn weighted_index_should_keep_total_weight_at_failed_update() {
        let weights = vec![Number(1.0), Number(2.0), Number(3.0)];
        let mut weighted_index = WeightedIndex::new(weights).unwrap();
        assert!(
            weighted_index
                .update(&[(0, &Number(0.0)), (1, &Number(-1.0))])
                .is_err(),
        );
        weighted_index.update(&[(2, &Number(2.0))]).unwrap();
        let rng = rand::thread_rng();
        let indices = rng
            .sample_iter(&weighted_index)
            .take(10)
            .collect::<Vec<usize>>();
        assert_eq!(indices, &[
            0, 0,
            1, 1, 1, 1,
            2, 2, 2, 2,
        ]);
    }
}
