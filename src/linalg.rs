//! Provides basic linear algebra operations.

use num_traits::Float;

/// Calculates a dot (inner) product of given two vectors.
pub fn dot<T>(xs: &[T], ys: &[T]) -> T
where
    T: Float + std::iter::Sum + std::ops::AddAssign,
{
    assert_eq!(xs.len(), ys.len());
    const C: usize = 16;
    let r = xs.len() % C;
    let mut ans = T::zero();
    if r != 0 {
        ans = xs[0..r].iter().zip(&ys[0..r]).map(|(x, y)| *x * *y).sum();
    }
    if r < xs.len() {
        let mut acc = [T::zero(); C];
        let mut i = r;
        while i + C <= xs.len() {
            let xs = &xs[i..i+C];
            let ys = &ys[i..i+C];
            for j in 0..C {
                acc[j] += xs[j] * ys[j];
            }
            i += C;
        }
        for v in &acc {
            ans += *v;
        }
    }
    ans
}

/// Calculates the Euclidean norm of a given vector.
///
/// Returns zero if the vector is empty.
pub fn norm2<T>(xs: &[T]) -> T
where
    T: Float + std::iter::Sum + std::ops::AddAssign,
{
    let mx = max_abs(xs);
    if let Some(mx) = mx {
        let mx_sqrt = mx.sqrt();
        norm2_scaled(xs, T::one() / mx_sqrt) * mx_sqrt
    } else {
        T::zero()
    }
}

/// Calculates the Euclidean norm of a scaled vector.
fn norm2_scaled<T>(xs: &[T], a: T) -> T
where
    T: Float + std::iter::Sum + std::ops::AddAssign,
{
    const C: usize = 16;
    let r = xs.len() % C;
    let mut ans = T::zero();
    if r != 0 {
        ans = xs[..r].iter().map(|x| *x * a * *x * a).sum();
    }
    if r < xs.len() {
        let mut acc = [T::zero(); C];
        let xs = &xs[r..];
        let mut i = 0;
        while i + C <= xs.len() {
            let xs_c = &xs[i..i+C];
            for j in 0..C {
                let scaled = a * xs_c[j];
                acc[j] += scaled * scaled;
            }
            i += C;
        }
        for a in &acc {
            ans += *a;
        }
    }
    ans.sqrt()
}

/// Multiplies a scalar value to a given vector in place.
pub fn scale_in<T>(xs: &mut [T], a: T)
where
    T: Copy + std::ops::MulAssign,
{
    const C: usize = 16;
    let r = xs.len() % C;
    if r != 0 {
        for i in 0..r {
            xs[i] *= a;
        }
    }
    let xs = &mut xs[r..];
    let mut i = 0;
    while i + C <= xs.len() {
        let xs_c = &mut xs[i..i+C];
        for j in 0..C {
            xs_c[j] *= a;
        }
        i += C;
    }
}

/// Locates the maximum absoulte value in a given vector.
pub fn max_abs<T>(xs: &[T]) -> Option<T>
where
    T: Float,
{
    const C: usize = 16;
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
    T: Float,
{
    if xs.len() > 0 {
        let mut mx = xs[0];
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
