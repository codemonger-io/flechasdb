//! IO utilities.

use base64::{
    Engine,
    engine::general_purpose::{URL_SAFE_NO_PAD as base64_engine},
};
use std::ffi::OsStr;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

use crate::error::Error;

/// Abstracts a file system.
pub trait FileSystem {
    /// File whose name will be the hash of its contents.
    type HashedFileOut: HashedFileOut;
    /// File whose name is the hash of its contents.
    type HashedFileIn: HashedFileIn;

    /// Creates a file whose name will be the hash of its contents.
    fn create_hashed_file(&self) -> Result<Self::HashedFileOut, Error>;

    /// Creates a hashed file in a given directory.
    fn create_hashed_file_in<P>(
        &self,
        path: P,
    ) -> Result<Self::HashedFileOut, Error>
    where
        P: AsRef<Path>;

    /// Opens a file whose name is the hash of its contents.
    fn open_hashed_file<P>(
        &self,
        path: P,
    ) -> Result<Self::HashedFileIn, Error>
    where
        P: AsRef<Path>;
}

/// File whose name will be the hash of its contents.
pub trait HashedFileOut: Write {
    /// Persists the file.
    ///
    /// Finishes the calculation of the hash and persists the file.
    /// You should flush the stream before calling this function.
    ///
    /// Returns the encoded hash value that is supposed to be a URS-safe Base64
    /// encoded SHA256 digest.
    fn persist<S>(self, extension: S) -> Result<String, Error>
    where
        S: AsRef<OsStr>;
}

/// File whose name is the hash of its contents.
pub trait HashedFileIn: Read {
    /// Verifies the file.
    ///
    /// Finishes the calculation of the hash and verifies the file.
    /// You should call this function after the entire file has been read.
    ///
    /// File name is supposed to be a Base64 encoded URL-safe SHA256 digest.
    fn verify(self) -> Result<(), Error>;
}

/// File system uses the local file system.
pub struct LocalFileSystem {
    // Base path.
    base_path: PathBuf,
}

impl LocalFileSystem {
    /// Creates a local file system working under a given base path.
    pub fn new<P>(base_path: P) -> Self
    where
        P: AsRef<Path>,
    {
        Self {
            base_path: base_path.as_ref().to_path_buf(),
        }
    }
}

impl FileSystem for LocalFileSystem {
    type HashedFileOut = LocalHashedFileOut;
    type HashedFileIn = LocalHashedFileIn;

    fn create_hashed_file(&self) -> Result<Self::HashedFileOut, Error> {
        LocalHashedFileOut::create(self.base_path.clone())
    }

    fn create_hashed_file_in<P>(
        &self,
        path: P,
    ) -> Result<Self::HashedFileOut, Error>
    where
        P: AsRef<Path>,
    {
        LocalHashedFileOut::create(self.base_path.join(path))
    }

    fn open_hashed_file<P>(
        &self,
        path: P,
    ) -> Result<Self::HashedFileIn, Error>
    where
        P: AsRef<Path>,
    {
        LocalHashedFileIn::open(self.base_path.join(path))
    }
}

/// Writable file in the local file system.
///
/// Created as a temporary file and renamed to the hash of its contents.
pub struct LocalHashedFileOut {
    // Temporary file.
    tempfile: NamedTempFile,
    // Persisted path.
    base_path: PathBuf,
    // Context to calculate an SHA-256 digest.
    context: ring::digest::Context,
}

impl LocalHashedFileOut {
    /// Creates a temporary file to be persisted under a given path.
    fn create(base_path: PathBuf) -> Result<Self, Error> {
        let tempfile = NamedTempFile::new()?;
        Ok(LocalHashedFileOut {
            tempfile,
            base_path,
            context: ring::digest::Context::new(&ring::digest::SHA256),
        })
    }
}

impl Write for LocalHashedFileOut {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.context.update(buf);
        self.tempfile.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.tempfile.flush()
    }
}

impl HashedFileOut for LocalHashedFileOut {
    fn persist<S>(mut self, extension: S) -> Result<String, Error>
    where
        S: AsRef<OsStr>,
    {
        self.flush()?;
        if !self.base_path.exists() {
            std::fs::create_dir_all(&self.base_path)?;
        }
        let hash = self.context.finish();
        let hash = base64_engine.encode(&hash);
        let path = self.base_path.join(&hash).with_extension(extension);
        self.tempfile.persist(path)?;
        Ok(hash)
    }
}

/// Readable file in the local file system.
pub struct LocalHashedFileIn {
    file: std::fs::File,
    path: PathBuf,
    // Context to calculate an SHA-256 digest.
    context: ring::digest::Context,
}

impl LocalHashedFileIn {
    /// Opens a file whose name is the hash of its contents.
    fn open(path: PathBuf) -> Result<Self, Error> {
        let file = std::fs::File::open(&path)?;
        Ok(LocalHashedFileIn {
            file,
            path,
            context: ring::digest::Context::new(&ring::digest::SHA256),
        })
    }
}

impl Read for LocalHashedFileIn {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.file.read(buf)?;
        self.context.update(&buf[..n]);
        Ok(n)
    }
}

impl HashedFileIn for LocalHashedFileIn {
    fn verify(self) -> Result<(), Error> {
        let hash = self.context.finish();
        let hash = base64_engine.encode(&hash);
        if hash.as_str() == self.path.file_stem().unwrap_or(OsStr::new("")) {
            Ok(())
        } else {
            Err(Error::VerificationFailure(format!(
                "Expected hash {:?}, but got {}",
                self.path.file_stem(),
                hash,
            )))
        }
    }
}
