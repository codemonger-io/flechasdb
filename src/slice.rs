/// Represents a type that can be referenced as a slice.
pub trait AsSlice<T> {
    /// Returns a reference to the underlying slice.
    fn as_slice(&self) -> &[T];
}

impl<T> AsSlice<T> for [T] {
    fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T> AsSlice<T> for &[T] {
    fn as_slice(&self) -> &[T] {
        *self
    }
}

impl<T> AsSlice<T> for Vec<T> {
    fn as_slice(&self) -> &[T] {
        self.as_slice()
    }
}
