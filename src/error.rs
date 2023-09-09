/// Error type for flechasdb.
#[derive(Debug)]
pub enum Error {
    /// Invalid arguments.
    InvalidArgs(String),
    /// I/O error.
    IOError(std::io::Error),
    /// Error on `protobuf`.
    ProtobufError(protobuf::Error),
}

impl Error {
    /// Converts into a string.
    pub fn to_string(self) -> String {
        match self {
            Self::InvalidArgs(s) => s,
            Self::IOError(e) => format!("I/O error: {}", e),
            Self::ProtobufError(e) => format!("Protobuf error: {}", e),
        }
    }
}

impl std::error::Error for Error {}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
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
