//! Common error type for flechasdb.

/// Common error type for flechasdb.
#[derive(Debug)]
pub enum Error {
    /// Invalid arguments.
    InvalidArgs(String),
    /// Invalid data.
    InvalidData(String),
    /// Invalid context.
    InvalidContext(String),
    /// Verification has failed.
    VerificationFailure(String),
    /// I/O error.
    IOError(std::io::Error),
    /// Error on `protobuf`.
    ProtobufError(protobuf::Error),
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgs(s) |
            Self::InvalidData(s) |
            Self::InvalidContext(s) |
            Self::VerificationFailure(s) => write!(f, "{}", s),
            Self::IOError(e) => write!(f, "I/O error: {}", e),
            Self::ProtobufError(e) => write!(f, "Protobuf error: {}", e),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::IOError(e)
    }
}

impl From<protobuf::Error> for Error {
    fn from(e: protobuf::Error) -> Self {
        Self::ProtobufError(e)
    }
}

impl From<tempfile::PersistError> for Error {
    fn from(e: tempfile::PersistError) -> Self {
        Self::IOError(e.error)
    }
}

impl From<tempfile::PathPersistError> for Error {
    fn from(e: tempfile::PathPersistError) -> Self {
        Self::IOError(std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
