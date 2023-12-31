//! Provides basic linear algebra operations.

use core::ops::{AddAssign, Div, Mul, MulAssign, Sub, SubAssign};

use crate::numbers::{Abs, One, Sqrt, Zero};

const UNROLL: usize = 16;

/// Calculates the dot (inner) product of given two vectors.
///
/// Unrolls loops to facilitate vectorization.
pub fn dot<T>(xs: &[T], ys: &[T]) -> T
where
    T: Zero + AddAssign + Mul<Output = T> + Copy,
{
    assert_eq!(xs.len(), ys.len());
    const C: usize = UNROLL;
    if xs.len() < C {
        return dot_naive(xs, ys);
    }
    let mut acc = [T::zero(); C];
    let r = xs.len() % C;
    if r != 0 {
        for i in 0..r {
            acc[i] = xs[i] * ys[i];
        }
    }
    let xs = &xs[r..];
    let ys = &ys[r..];
    let mut i = 0;
    while i + C <= xs.len() {
        let xs = &xs[i..i+C];
        let ys = &ys[i..i+C];
        for j in 0..C {
            acc[j] += xs[j] * ys[j];
        }
        i += C;
    }
    sum_naive(&acc[..])
}

/// Calculates the dot (inner) product of given two vectors.
pub fn dot_naive<T>(xs: &[T], ys: &[T]) -> T
where
    T: Zero + AddAssign + Mul<Output = T> + Copy,
{
    assert_eq!(xs.len(), ys.len());
    let mut ans = T::zero();
    for i in 0..xs.len() {
        ans += xs[i] * ys[i];
    }
    ans
}

/// Calculates the Euclidean norm of a given vector.
///
/// This function is safe if `xs` contains an extermely large or small value
/// whose square overflows or underflows.
///
/// Returns zero if the vector is empty.
pub fn norm2<T>(xs: &[T]) -> T
where
    T: Abs + One + Sqrt + Zero + AddAssign + Div<Output = T> + Mul<Output = T> + PartialOrd + PartialEq + Copy,
{
    let mx = max_abs(xs);
    if let Some(mx) = mx {
        if mx == T::zero() {
            return T::zero();
        }
        let mx_sqrt = mx.sqrt();
        norm2_scaled(xs, T::one() / mx_sqrt) * mx_sqrt
    } else {
        T::zero()
    }
}

// Calculates the Euclidean norm of a scaled vector.
fn norm2_scaled<T>(xs: &[T], a: T) -> T
where
    T: Sqrt + Zero + AddAssign + Mul<Output = T> + Copy,
{
    const C: usize = UNROLL;
    if xs.len() < C {
        return norm2_scaled_naive(xs, a);
    }
    let mut acc = [T::zero(); C];
    let r = xs.len() % C;
    if r != 0 {
        for i in 0..r {
            let scaled = a * xs[i];
            acc[i] += scaled * scaled;
        }
    }
    let xs = &xs[r..];
    let mut i = 0;
    while i + C <= xs.len() {
        let xs = &xs[i..i+C];
        for j in 0..C {
            let scaled = a * xs[j];
            acc[j] += scaled * scaled;
        }
        i += C;
    }
    sum_naive(&acc[..]).sqrt()
}

/// Calculates the Euclidean norm of a scaled vector.
fn norm2_scaled_naive<T>(xs: &[T], a: T) -> T
where
    T: Sqrt + Zero + AddAssign + Mul<Output = T> + Copy,
{
    let mut acc = T::zero();
    for x in xs {
        let scaled = *x * a;
        acc += scaled * scaled;
    }
    acc.sqrt()
}

/// Calculates the Euclidean norm of a given vector.
pub fn norm2_naive<T>(xs: &[T]) -> T
where
    T: Sqrt + Zero + AddAssign + Mul<Output = T> + Copy,
{
    dot_naive(xs, xs).sqrt()
}

/// Calculates the Euclidean norm of a given vector.
///
/// This function will produce a more accurate result if `xs` contains an
/// extermely large or small value.
pub fn norm2_naive_check<T>(xs: &[T]) -> T
where
    T: Abs + One + Sqrt + Zero + AddAssign + Div<Output = T> + Mul<Output = T> + PartialOrd + PartialEq + Copy,
{
    let mx = max_abs_naive(xs);
    if let Some(mx) = mx {
        if mx == T::zero() {
            return T::zero();
        }
        let mx_sqrt = mx.sqrt();
        norm2_scaled_naive(xs, T::one() / mx_sqrt) * mx_sqrt
    } else {
        T::zero()
    }
}

/// Adds a vector to another vector in place.
pub fn add_in<T>(ls: &mut [T], rs: &[T])
where
    T: AddAssign + Copy,
{
    assert_eq!(ls.len(), rs.len());
    ls.iter_mut().zip(rs).for_each(|(l, r)| *l += *r);
}

/// Subtracts a vector from another vector.
pub fn subtract<T>(ls: &[T], rs: &[T], out: &mut [T])
where
    T: Sub<Output = T> + Copy,
{
    assert_eq!(ls.len(), rs.len());
    assert_eq!(ls.len(), out.len());
    ls.iter().zip(rs).zip(out).for_each(|((l, r), o)| *o = *l - *r);
}

/// Subtracts a vector from another vector in place.
pub fn subtract_in<T>(ls: &mut [T], rs: &[T])
where
    T: SubAssign + Copy,
{
    assert_eq!(ls.len(), rs.len());
    ls.iter_mut().zip(rs).for_each(|(l, r)| *l -= *r);
}

/// Subtracts a vector from another vector in place.
pub fn subtract_in_naive<T>(ls: &mut [T], rs: &[T])
where
    T: SubAssign + Copy,
{
    assert_eq!(ls.len(), rs.len());
    for i in 0..ls.len() {
        ls[i] -= rs[i];
    }
}

/// Multiplies a scalar value to a given vector in place.
pub fn scale_in<T>(xs: &mut [T], a: T)
where
    T: MulAssign + Copy,
{
    xs.iter_mut().for_each(|x| *x *= a);
}

/// Multiplies a scalar value to a given vector in place.
pub fn scale_in_naive<T>(xs: &mut [T], a: T)
where
    T: MulAssign + Copy,
{
    for x in xs {
        *x *= a;
    }
}

/// Sums all the elements in a give vector.
///
/// Unrolls loops to facilitate vectorization.
pub fn sum<T>(xs: &[T]) -> T
where
    T: Zero + AddAssign + Copy,
{
    const C: usize = UNROLL;
    if xs.len() < C {
        return sum_naive(xs);
    }
    let mut acc: Vec<T> = Vec::with_capacity(C);
    acc.extend_from_slice(&xs[0..C]);
    let xs = &xs[C..];
    let r = xs.len() % C;
    if r != 0 {
        for i in 0..r {
            acc[i] += xs[i];
        }
    }
    let xs = &xs[r..];
    let mut i = 0;
    while i + C <= xs.len() {
        let xs = &xs[i..i+C];
        for i in 0..C {
            acc[i] += xs[i];
        }
        i += C;
    }
    sum_naive(&acc[..])
}

/// Sums all the elements in a given vector.
pub fn sum_naive<T>(xs: &[T]) -> T
where
    T: Zero + AddAssign + Copy,
{
    let mut ans = T::zero();
    for x in xs {
        ans += *x;
    }
    ans
}

/// Locates the minimum value in a given vector.
///
/// Unrolls loops to facilitate vectorization.
pub fn min<T>(xs: &[T]) -> Option<T>
where
    T: PartialOrd + Copy,
{
    const C: usize = UNROLL;
    if xs.len() < C {
        return min_naive(xs);
    }
    let mut acc: Vec<T> = Vec::with_capacity(C);
    acc.extend_from_slice(&xs[0..C]);
    let xs = &xs[C..];
    let r = xs.len() % C;
    if r != 0 {
        for i in 0..r {
            if xs[i] < acc[i] {
                acc[i] = xs[i];
            }
        }
    }
    let xs = &xs[r..];
    let mut i = 0;
    while i + C <= xs.len() {
        let c_xs = &xs[i..i+C];
        for i in 0..C {
            if c_xs[i] < acc[i] {
                acc[i] = c_xs[i];
            }
        }
        i += C;
    }
    min_naive(&acc)
}

/// Locates the minimum value in a given vector.
pub fn min_naive<T>(xs: &[T]) -> Option<T>
where
    T: PartialOrd + Copy,
{
    if xs.len() > 0 {
        let mut mn = xs[0];
        for i in 1..xs.len() {
            if xs[i] < mn {
                mn = xs[i];
            }
        }
        Some(mn)
    } else {
        None
    }
}

/// Locates the maximum absoulte value in a given vector.
///
/// Unrolls loops to facilitate vectorization.
pub fn max_abs<T>(xs: &[T]) -> Option<T>
where
    T: Abs + PartialOrd + Copy,
{
    const C: usize = UNROLL;
    if xs.len() < C {
        return max_abs_naive(xs);
    }
    let mut acc: Vec<T> = Vec::with_capacity(C);
    for i in 0..C {
        acc.push(xs[i].abs());
    }
    let xs = &xs[C..];
    let r = xs.len() % C;
    if r != 0 {
        for i in 0..r {
            if xs[i].abs() > acc[i] {
                acc[i] = xs[i].abs();
            }
        }
    }
    let xs = &xs[r..];
    let mut i = 0;
    while i + C <= xs.len() {
        let c_xs = &xs[i..i+C];
        for i in 0..C {
            if c_xs[i].abs() > acc[i] {
                acc[i] = c_xs[i].abs();
            }
        }
        i += C;
    }
    let mut mx = acc[0];
    for i in 1..C {
        if mx < acc[i] {
            mx = acc[i];
        }
    }
    Some(mx)
}

/// Locates the maximum absolute value in a given vector.
pub fn max_abs_naive<T>(xs: &[T]) -> Option<T>
where
    T: Abs + PartialOrd + Copy,
{
    if xs.len() > 0 {
        let mut mx = xs[0].abs();
        for i in 1..xs.len() {
            if xs[i].abs() > mx {
                mx = xs[i].abs();
            }
        }
        Some(mx)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // compares floating point numbers.
    macro_rules! assert_eq_f {
        ($actual:expr, $expected:expr, $err:expr) => {
            match (&$actual, &$expected, &$err) {
                (actual_val, expected_val, err_val) => {
                    assert!(
                        (*expected_val - *actual_val).abs() < *err_val,
                        "expected {} but got {}",
                        *expected_val,
                        *actual_val,
                    );
                }
            }
        };
    }

    #[test]
    fn dot_should_calculate_inner_product_of_one_element_vectors() {
        let xs: &[f32] = &[2.0];
        let ys: &[f32] = &[3.0];
        assert_eq!(dot(xs, ys), 6.0);
    }

    #[test]
    fn dot_should_calculate_inner_product_of_16_element_vectors() {
        let xs: &[f32] = &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            2.0, 4.0, 6.0, 8.0,
            -1.0, -2.0, -3.0, -4.0,
        ];
        let ys: &[f32] = &[
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            -1.0, -1.0, -1.0, -1.0,
            1.0, 2.0, 3.0, 4.0,
        ];
        assert_eq!(
            dot(xs, ys),
            1.0 + 2.0 + 3.0 + 4.0
            + 10.0 + 12.0 + 14.0 + 16.0
            - 2.0 - 4.0 - 6.0 - 8.0
            - 1.0 - 4.0 - 9.0 - 16.0,
        );
    }

    #[test]
    fn dot_should_calculate_inner_product_of_32_element_vectors() {
        let xs: &[f32] = &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            2.0, 4.0, 6.0, 8.0,
            -1.0, -2.0, -3.0, -4.0,
            1.0, 2.0, 3.0, 5.0,
            7.0, 11.0, 13.0, 17.0,
            -1.0, -2.0, -3.0, -5.0,
            -7.0, -11.0, -13.0, -17.0,
        ];
        let ys: &[f32] = &[
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            -1.0, -1.0, -1.0, -1.0,
            1.0, 2.0, 3.0, 4.0,
            4.0, 3.0, 2.0, 1.0,
            -1.0, -2.0, -3.0, -4.0,
            3.0, 3.0, 3.0, 3.0,
            -1.0, 1.0, -1.0, 1.0,
        ];
        assert_eq!(
            dot(xs, ys),
            1.0 + 2.0 + 3.0 + 4.0
            + 10.0 + 12.0 + 14.0 + 16.0
            - 2.0 - 4.0 - 6.0 - 8.0
            - 1.0 - 4.0 - 9.0 - 16.0
            + 4.0 + 6.0 + 6.0 + 5.0
            - 7.0 - 22.0 - 39.0 - 68.0
            - 3.0 - 6.0 - 9.0 - 15.0
            + 7.0 - 11.0 + 13.0 - 17.0,
        );
    }

    #[test]
    fn dot_should_calculate_inner_product_of_33_element_vectors() {
        let xs: &[f32] = &[
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            2.0, 4.0, 6.0, 8.0,
            -1.0, -2.0, -3.0, -4.0,
            1.0, 2.0, 3.0, 5.0,
            7.0, 11.0, 13.0, 17.0,
            -1.0, -2.0, -3.0, -5.0,
            -7.0, -11.0, -13.0, -17.0,
            10.0,
        ];
        let ys: &[f32] = &[
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            -1.0, -1.0, -1.0, -1.0,
            1.0, 2.0, 3.0, 4.0,
            4.0, 3.0, 2.0, 1.0,
            -1.0, -2.0, -3.0, -4.0,
            3.0, 3.0, 3.0, 3.0,
            -1.0, 1.0, -1.0, 1.0,
            5.0,
        ];
        assert_eq!(
            dot(xs, ys),
            1.0 + 2.0 + 3.0 + 4.0
            + 10.0 + 12.0 + 14.0 + 16.0
            - 2.0 - 4.0 - 6.0 - 8.0
            - 1.0 - 4.0 - 9.0 - 16.0
            + 4.0 + 6.0 + 6.0 + 5.0
            - 7.0 - 22.0 - 39.0 - 68.0
            - 3.0 - 6.0 - 9.0 - 15.0
            + 7.0 - 11.0 + 13.0 - 17.0
            + 50.0,
        );
    }

    #[test]
    fn dot_should_return_zero_for_empty_vectors() {
        let v: &[f32] = &[];
        assert_eq!(dot(v, v), 0.0);
    }

    #[test]
    fn norm2_should_calculate_norm_of_one_element_vector() {
        let v: &[f32] = &[2.0];
        assert_eq_f!(norm2(v), 2.0, 1.0e-6);
    }

    #[test]
    fn norm2_should_calculate_norm_of_16_element_vector() {
        let v: &[f32] = &[
            1.0, 2.0, 3.0, 4.0,
            -1.0, -2.0, -3.0, -4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ]; // sum(x^2) = 680
        assert_eq_f!(norm2(v), 26.07681, 1.0e-5);
    }

    #[test]
    fn norm2_should_calculate_norm_of_32_element_vector() {
        let v: &[f32] = &[
            1.0, 2.0, 3.0, 4.0,
            -1.0, -2.0, -3.0, -4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            -4.0, 2.0, -4.0, 2.0,
            1.0, 2.0, 3.0, 5.0,
            7.0, 11.0, 13.0, 17.0,
            3.0, 6.0, 9.0, 12.0,
        ]; // sum(x^2) = 1657
        assert_eq_f!(norm2(v), 40.70626, 1.0e-5);
    }

    #[test]
    fn norm2_should_calculate_norm_of_33_element_vector() {
        let v: &[f32] = &[
            1.0, 2.0, 3.0, 4.0,
            -1.0, -2.0, -3.0, -4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            -4.0, 2.0, -4.0, 2.0,
            1.0, 2.0, 3.0, 5.0,
            7.0, 11.0, 13.0, 17.0,
            3.0, 6.0, 9.0, 12.0,
            13.0,
        ]; // sum(x^2) = 1826
        assert_eq_f!(norm2(v), 42.73172, 1.0e-5);
    }

    #[test]
    fn norm2_should_return_zero_for_empty_vector() {
        let v: &[f32] = &[];
        assert_eq!(norm2(v), 0.0);
    }

    #[test]
    fn norm2_should_return_zero_for_zero_vector() {
        let v: &[f32] = &[0.0];
        assert_eq!(norm2(v), 0.0);
        let v: &[f32] = &[0.0, 0.0, 0.0];
        assert_eq!(norm2(v), 0.0);
    }

    #[test]
    fn norm2_should_calculate_norm_of_vector_of_one_extremely_large_value() {
        let v: &[f32] = &[1.0e+36];
        assert_eq_f!(norm2(v), 1.0e+36, 1.0e+30);
    }

    #[test]
    fn norm2_should_calculate_norm_of_vector_of_16_exteremely_large_values() {
        let v: &[f32] = &[1.0e+36; 16];
        assert_eq_f!(norm2(v), 4.0e+36, 1.0e+31);
    }

    #[test]
    fn norm2_should_calculate_norm_of_vector_of_one_extremely_small_positive_value() {
        let v: &[f32] = &[1.0e-30];
        assert_eq_f!(norm2(v), 1.0e-30, 1.0e-36);
    }

    #[test]
    fn norm2_should_caclulate_norm_of_vector_of_16_extremely_small_positive_values() {
        let v: &[f32] = &[1.0e-30; 16];
        assert_eq_f!(norm2(v), 4.0e-30, 1.0e-35);
    }

    #[test]
    fn add_in_should_add_one_element_vectors() {
        let xs: &mut [f32] = &mut [1.0];
        let ys: &[f32] = &[2.0];
        add_in(xs, ys);
        assert_eq!(xs, &[3.0]);
    }

    #[test]
    fn add_in_should_add_three_element_vectors() {
        let xs: &mut [f32] = &mut [0.0, -1.0, 2.0];
        let ys: &[f32] = &[-1.0, 2.0, -3.0];
        add_in(xs, ys);
        assert_eq!(xs, &[-1.0, 1.0, -1.0]);
    }

    #[test]
    fn add_in_should_work_with_empty_vectors() {
        let xs: &mut [f32] = &mut [];
        let ys: &[f32] = &[];
        add_in(xs, ys);
        assert_eq!(xs, &[]);
    }

    #[test]
    fn subtract_should_subtract_one_element_vectors() {
        let xs: &[f32] = &[1.0];
        let ys: &[f32] = &[2.0];
        let mut out = [0.0f32];
        subtract(xs, ys, &mut out);
        assert_eq!(&out, &[-1.0]);
    }

    #[test]
    fn subtract_should_subtract_three_element_vectors() {
        let xs: &[f32] = &[0.0, -1.0, 2.0];
        let ys: &[f32] = &[-1.0, 2.0, -3.0];
        let mut out = [0.0f32; 3];
        subtract(xs, ys, &mut out);
        assert_eq!(&out, &[1.0, -3.0, 5.0]);
    }

    #[test]
    fn subtract_should_work_with_empty_vectors() {
        let xs: &[f32] = &[];
        let ys: &[f32] = &[];
        let mut out = [0.0f32; 0];
        subtract(xs, ys, &mut out);
        assert_eq!(&out, &[]);
    }

    #[test]
    fn subtract_in_should_subtract_one_element_vectors() {
        let xs: &mut [f32] = &mut [1.0];
        let ys: &[f32] = &[2.0];
        subtract_in(xs, ys);
        assert_eq!(xs, &[-1.0]);
    }

    #[test]
    fn subtract_in_should_subtract_three_element_vectors() {
        let xs: &mut [f32] = &mut [0.0, -1.0, 2.0];
        let ys: &[f32] = &[-1.0, 2.0, -3.0];
        subtract_in(xs, ys);
        assert_eq!(xs, &[1.0, -3.0, 5.0]);
    }

    #[test]
    fn subtract_in_should_work_with_empty_vectors() {
        let xs: &mut [f32] = &mut [];
        let ys: &[f32] = &[];
        subtract_in(xs, ys);
        assert_eq!(xs, &[]);
    }

    #[test]
    fn scale_in_should_scale_one_element_vector() {
        let xs: &mut [f32] = &mut [2.0];
        scale_in(xs, 3.0);
        assert_eq!(xs, &[6.0]);
    }

    #[test]
    fn scale_in_should_scale_three_element_vector() {
        let xs: &mut [f32] = &mut [1.0, -2.0, 3.0];
        scale_in(xs, 5.0);
        assert_eq!(xs, &[5.0, -10.0, 15.0]);
    }

    #[test]
    fn scale_in_should_work_with_empty_vector() {
        let xs: &mut [f32] = &mut [];
        scale_in(xs, 2.0);
        assert_eq!(xs, &[]);
    }

    #[test]
    fn sum_should_calculate_total_of_one_element() {
        let v: &[f32] = &[3.0];
        assert_eq!(sum(&v), 3.0);
    }

    #[test]
    fn sum_should_calculate_total_of_16_elements() {
        let v: &[f32] = &[
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
            5.0, 10.0, 15.0, 20.0,
            -1.0, -2.0, -3.0, -4.0,
        ];
        assert_eq!(sum(&v), 70.0);
    }

    #[test]
    fn sum_should_calculate_total_of_32_elements() {
        let v: &[f32] = &[
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
            5.0, 10.0, 15.0, 20.0,
            -1.0, -2.0, -3.0, -4.0,
            0.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 7.0,
            8.0, 9.0, 10.0, 11.0,
            12.0, 13.0, 14.0, 15.0,
        ];
        assert_eq!(sum(&v), 190.0);
    }

    #[test]
    fn sum_should_calculate_total_of_33_elements() {
        let v: &[f32] = &[
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
            5.0, 10.0, 15.0, 20.0,
            -1.0, -2.0, -3.0, -4.0,
            0.0, 1.0, 2.0, 3.0,
            4.0, 5.0, 6.0, 7.0,
            8.0, 9.0, 10.0, 11.0,
            12.0, 13.0, 14.0, 15.0,
            -1.0,
        ];
        assert_eq!(sum(&v), 189.0);
    }

    #[test]
    fn sum_should_return_zero_for_empty_slice() {
        let v: &[f32] = &[];
        assert_eq!(sum(&v), 0.0);
    }

    #[test]
    fn min_should_return_value_in_one_element_vector() {
        let v: &[f32] = &[1.0];
        assert_eq!(min(&v), Some(1.0));
    }

    #[test]
    fn min_should_return_minimum_in_16_element_vector() {
        let v: &[f32] = &[
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -4.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(min(&v), Some(-4.0));
    }

    #[test]
    fn min_should_return_minimum_in_32_element_vector() {
        let v: &[f32] = &[
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, -5.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(min(&v), Some(-5.0));
    }

    #[test]
    fn min_should_return_minimum_in_35_element_vector() {
        let v: &[f32] = &[
            0.0, -2.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        assert_eq!(min(&v), Some(-2.0));
    }

    #[test]
    fn min_should_return_none_for_empty_vector() {
        let v: &[f32] = &[];
        assert_eq!(min(&v), None);
    }

    #[test]
    fn max_abs_should_return_absolute_value_in_one_element_vector() {
        let v: &[f32] = &[1.0];
        assert_eq!(max_abs(&v), Some(1.0));
        let v: &[f32] = &[-1.0];
        assert_eq!(max_abs(&v), Some(1.0));
    }

    #[test]
    fn max_abs_should_return_maximum_absolute_in_16_element_vector() {
        let v: &[f32] = &[
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 3.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(max_abs(&v), Some(3.0));
        let v: &[f32] = &[
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -2.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(max_abs(&v), Some(2.0));
    }

    #[test]
    fn max_abs_should_return_maximum_absolute_in_32_element_vector() {
        let v: &[f32] = &[
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 4.0, 0.0,
        ];
        assert_eq!(max_abs(&v), Some(4.0));
        let v: &[f32] = &[
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, -3.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            -7.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_eq!(max_abs(&v), Some(7.0));
    }

    #[test]
    fn max_abs_should_return_maximum_absolute_in_35_element_vector() {
        let v: &[f32] = &[
            0.0, 6.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
        ];
        assert_eq!(max_abs(&v), Some(6.0));
        let v: &[f32] = &[
            0.0, 0.0, -9.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            -4.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -8.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ];
        assert_eq!(max_abs(&v), Some(9.0));
    }

    #[test]
    fn max_abs_should_return_none_for_empty_vector() {
        let v: &[f32] = &[];
        assert_eq!(max_abs(&v), None);
    }
}
