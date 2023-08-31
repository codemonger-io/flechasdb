/// Error type for flechasdb.
#[derive(Debug)]
pub enum Error {
    /// Invalid arguments.
    InvalidArgs(String),
}

impl Error {
    /// Converts into a string.
    pub fn to_string(self) -> String {
        match self {
            Self::InvalidArgs(s) => s,
        }
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}
