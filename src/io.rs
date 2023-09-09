//! IO utilities.

use base64::{
    Engine,
    engine::general_purpose::{URL_SAFE_NO_PAD as base64_engine},
};
use std::ffi::OsStr;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

use crate::error::Error;

/// Abstracts a file system.
pub trait FileSystem {
    /// File whose name will be the hash of its contents.
    type HashedFile: HashedFile;

    /// Creates a file whose name will be the hash of its contents.
    fn create_hashed_file(&mut self) -> Result<Self::HashedFile, Error>;

    /// Creates a hashed file in a given directory.
    fn create_hashed_file_in<P>(
        &mut self,
        path: P,
    ) -> Result<Self::HashedFile, Error>
    where
        P: AsRef<Path>;
}

/// File whose name will be the hash of its contents.
pub trait HashedFile: Write {
    /// Persists the file.
    ///
    /// Finishes the calculation of the hash and persists the file.
    /// You should flush the stream before calling this function.
    ///
    /// Returns the encoded hash value that is supposed to be Base64 encoded
    /// SHA256 digest.
    fn persist<S>(self, extension: S) -> Result<String, Error>
    where
        S: AsRef<OsStr>;
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
    type HashedFile = LocalHashedFile;

    fn create_hashed_file(&mut self) -> Result<Self::HashedFile, Error> {
        LocalHashedFile::create(self.base_path.clone())
    }

    fn create_hashed_file_in<P>(
        &mut self,
        path: P,
    ) -> Result<Self::HashedFile, Error>
    where
        P: AsRef<Path>,
    {
        LocalHashedFile::create(self.base_path.join(path))
    }
}

/// File in the local file system.
///
/// Created as a temporary file and renamed to the hash of its contents.
pub struct LocalHashedFile {
    // Temporary file.
    tempfile: NamedTempFile,
    // Persisted path.
    base_path: PathBuf,
    // Context to calculate SHA-256 hash.
    context: ring::digest::Context,
}

impl LocalHashedFile {
    /// Creates a temporary file to be persisted under a given path.
    fn create(base_path: PathBuf) -> Result<Self, Error> {
        let tempfile = NamedTempFile::new()?;
        Ok(LocalHashedFile {
            tempfile,
            base_path,
            context: ring::digest::Context::new(&ring::digest::SHA256),
        })
    }
}

impl Write for LocalHashedFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.context.update(buf);
        self.tempfile.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.tempfile.flush()
    }
}

impl HashedFile for LocalHashedFile {
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
