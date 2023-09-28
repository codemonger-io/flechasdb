//! Provides traits for numbers.
//!
//! Focuses on floating point numbers so far.

/// Represents a number that has zero.
pub trait Zero {
    /// Returns zero.
    fn zero() -> Self;
}

impl Zero for f32 {
    fn zero() -> f32 {
        0.0f32
    }
}

impl Zero for f64 {
    fn zero() -> f64 {
        0.0f64
    }
}

/// Represents a number that has one.
pub trait One {
    /// Returns one.
    fn one() -> Self;
}

impl One for f32 {
    fn one() -> f32 {
        1.0f32
    }
}

impl One for f64 {
    fn one() -> f64 {
        1.0f64
    }
}

/// Represents a number that has positive infinity.
pub trait Infinity {
    /// Returns positive infinity.
    fn infinity() -> Self;
}

impl Infinity for f32 {
    fn infinity() -> f32 {
        f32::INFINITY
    }
}

impl Infinity for f64 {
    fn infinity() -> f64 {
        f64::INFINITY
    }
}

/// Represents a number converted from another primitive type.
pub trait FromAs<T> {
    /// Returns the converted number.
    fn from_as(t: T) -> Self;
}

impl FromAs<usize> for f32 {
    fn from_as(t: usize) -> f32 {
        t as f32
    }
}

impl FromAs<usize> for f64 {
    fn from_as(t: usize) -> f64 {
        t as f64
    }
}

/// Represents a number that can return an absolute value.
pub trait Abs {
    /// Returns the absolute value.
    fn abs(self) -> Self;
}

impl Abs for f32 {
    fn abs(self) -> f32 {
        self.abs()
    }
}

impl Abs for f64 {
    fn abs(self) -> f64 {
        self.abs()
    }
}

/// Represents a number that can calculate a square root.
pub trait Sqrt {
    /// Returns the square root.
    fn sqrt(self) -> Self;
}

impl Sqrt for f32 {
    fn sqrt(self) -> f32 {
        self.sqrt()
    }
}

impl Sqrt for f64 {
    fn sqrt(self) -> f64 {
        self.sqrt()
    }
}
