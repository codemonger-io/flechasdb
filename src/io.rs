//! IO utilities.

use base64::{
    Engine,
    engine::general_purpose::{URL_SAFE_NO_PAD as base64_engine},
};
use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tempfile::NamedTempFile;

use crate::error::Error;

/// Abstracts a file system.
pub trait FileSystem {
    /// File that calculates the hash of its contents.
    type HashedFileOut: HashedFileOut;
    /// File whose contents can be verified with the hash.
    type HashedFileIn: HashedFileIn;

    /// Creates a file that calculates the hash of its contents.
    fn create_hashed_file(&self) -> Result<Self::HashedFileOut, Error>;

    /// Creates a hashed file in a given directory.
    fn create_hashed_file_in(
        &self,
        path: impl AsRef<str>,
    ) -> Result<Self::HashedFileOut, Error>;

    /// Opens a file whose contents can be verified with a hash.
    fn open_hashed_file(
        &self,
        path: impl AsRef<str>,
    ) -> Result<Self::HashedFileIn, Error>;

    /// Creates a compressed file that calculates the hash of its contents.
    fn create_compressed_hashed_file(
        &self,
    ) -> Result<CompressedHashedFileOut<Self::HashedFileOut>, Error> {
        let file = self.create_hashed_file()?;
        Ok(CompressedHashedFileOut::new(file))
    }

    /// Creates a compressed hashed file in a given directory.
    fn create_compressed_hashed_file_in(
        &self,
        path: impl AsRef<str>,
    ) -> Result<CompressedHashedFileOut<Self::HashedFileOut>, Error> {
        let file = self.create_hashed_file_in(path)?;
        Ok(CompressedHashedFileOut::new(file))
    }

    /// Opens a compressed file whose contents can be verified with a hash.
    fn open_compressed_hashed_file(
        &self,
        path: impl AsRef<str>,
    ) -> Result<CompressedHashedFileIn<Self::HashedFileIn>, Error> {
        let file = self.open_hashed_file(path)?;
        Ok(CompressedHashedFileIn::new(file))
    }
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
    fn persist(self, extension: impl AsRef<str>) -> Result<String, Error>;
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

/// Compressed file that calculates the hash of its contents.
pub struct CompressedHashedFileOut<W>
where
    W: std::io::Write,
{
    encoder: ZlibEncoder<W>,
}

impl<W> CompressedHashedFileOut<W>
where
    W: std::io::Write,
{
    /// Writes compressed data to a given [`Write`].
    pub fn new(w: W) -> Self {
        Self {
            encoder: ZlibEncoder::new(w, Compression::default()),
        }
    }
}

impl<W> Write for CompressedHashedFileOut<W>
where
    W: std::io::Write,
{
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.encoder.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.encoder.flush()
    }
}

impl<W> HashedFileOut for CompressedHashedFileOut<W>
where
    W: HashedFileOut
{
    fn persist(self, extension: impl AsRef<str>) -> Result<String, Error> {
        self.encoder.finish()?.persist(extension)
    }
}

/// Compressed file whose contents can be verified with a hash.
pub struct CompressedHashedFileIn<R>
where
    R: std::io::Read,
{
    decoder: ZlibDecoder<R>,
}

impl<R> CompressedHashedFileIn<R>
where
    R: std::io::Read,
{
    /// Reads compressed data from a given [`Read`].
    pub fn new(r: R) -> Self {
        Self {
            decoder: ZlibDecoder::new(r),
        }
    }
}

impl<R> Read for CompressedHashedFileIn<R>
where
    R: std::io::Read,
{
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.decoder.read(buf)
    }
}

impl<R> HashedFileIn for CompressedHashedFileIn<R>
where
    R: HashedFileIn,
{
    fn verify(self) -> Result<(), Error> {
        self.decoder.into_inner().verify()
    }
}

/// File system uses the local file system.
pub struct LocalFileSystem {
    // Base path.
    base_path: PathBuf,
}

impl LocalFileSystem {
    /// Creates a local file system working under a given base path.
    pub fn new(base_path: impl AsRef<Path>) -> Self {
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

    fn create_hashed_file_in(
        &self,
        path: impl AsRef<str>,
    ) -> Result<Self::HashedFileOut, Error> {
        LocalHashedFileOut::create(self.base_path.join(path.as_ref()))
    }

    fn open_hashed_file(
        &self,
        path: impl AsRef<str>,
    ) -> Result<Self::HashedFileIn, Error> {
        LocalHashedFileIn::open(self.base_path.join(path.as_ref()))
    }
}

/// Writable file in the local file system.
///
/// Created as a temporary file and renamed to the hash of its contents.
pub struct LocalHashedFileOut {
    // Temporary file as a writer.
    tempfile: NamedTempFile,
    // Persisted path.
    base_path: PathBuf,
    // Context to calculate an SHA-256 digest.
    context: ring::digest::Context,
}

impl LocalHashedFileOut {
    /// Creates a temporary file to be persisted under a given path.
    ///
    /// The output is compressed with Zlib if `compress` is `true`.
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
    fn persist(mut self, extension: impl AsRef<str>) -> Result<String, Error> {
        self.tempfile.flush()?;
        if !self.base_path.exists() {
            std::fs::create_dir_all(&self.base_path)?;
        }
        let hash = self.context.finish();
        let hash = base64_engine.encode(&hash);
        let path = self.base_path
            .join(&hash)
            .with_extension(extension.as_ref());
        self.tempfile.persist(path)?;
        Ok(hash)
    }
}

/// Readable file in the local file system.
pub struct LocalHashedFileIn {
    file: File,
    path: PathBuf,
    // Context to calculate an SHA-256 digest.
    context: ring::digest::Context,
}

impl LocalHashedFileIn {
    /// Opens a file whose name is the hash of its contents.
    ///
    /// The file must be compressed with `zlib` if `comparessed` is `true`.
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
