//! IO utilities.

use base64::{
    Engine,
    engine::general_purpose::{URL_SAFE_NO_PAD as base64_engine},
};
use flate2::Compression;
use flate2::write::ZlibEncoder;
use flate2::read::ZlibDecoder;
use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use tempfile::{NamedTempFile, TempPath};

use crate::error::Error;

/// Abstracts a file system.
pub trait FileSystem {
    /// File whose name will be the hash of its contents.
    type HashedFileOut: HashedFileOut;
    /// File whose name is the hash of its contents.
    type HashedFileIn: HashedFileIn;

    /// Creates a file whose name will be the hash of its contents.
    fn create_hashed_file(
        &self,
        compress: bool,
    ) -> Result<Self::HashedFileOut, Error>;

    /// Creates a hashed file in a given directory.
    fn create_hashed_file_in(
        &self,
        path: impl AsRef<str>,
        compress: bool,
    ) -> Result<Self::HashedFileOut, Error>;

    /// Opens a file whose name is the hash of its contents.
    fn open_hashed_file(
        &self,
        path: impl AsRef<str>,
        compressed: bool,
    ) -> Result<Self::HashedFileIn, Error>;
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

    fn create_hashed_file(
        &self,
        compress: bool,
    ) -> Result<Self::HashedFileOut, Error> {
        LocalHashedFileOut::create(self.base_path.clone(), compress)
    }

    fn create_hashed_file_in(
        &self,
        path: impl AsRef<str>,
        compress: bool,
    ) -> Result<Self::HashedFileOut, Error> {
        LocalHashedFileOut::create(self.base_path.join(path.as_ref()), compress)
    }

    fn open_hashed_file(
        &self,
        path: impl AsRef<str>,
        compressed: bool,
    ) -> Result<Self::HashedFileIn, Error> {
        LocalHashedFileIn::open(self.base_path.join(path.as_ref()), compressed)
    }
}

/// Writable file in the local file system.
///
/// Created as a temporary file and renamed to the hash of its contents.
pub struct LocalHashedFileOut {
    // Temporary file path.
    temp_path: TempPath,
    // Underlying writer.
    writer: MaybeCompressedWrite<File>,
    // Persisted path.
    base_path: PathBuf,
    // Context to calculate an SHA-256 digest.
    context: ring::digest::Context,
}

// Maybe compressed `Write`.
enum MaybeCompressedWrite<W>
where
    W: std::io::Write,
{
    Compressed(ZlibEncoder<W>),
    Uncompressed(W),
}

impl<W> MaybeCompressedWrite<W>
where
    W: std::io::Write,
{
    fn finish(self) -> std::io::Result<W> {
        match self {
            MaybeCompressedWrite::Compressed(w) => w.finish(),
            MaybeCompressedWrite::Uncompressed(w) => Ok(w),
        }
    }
}

impl<W> Write for MaybeCompressedWrite<W>
where
    W: std::io::Write,
{
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            MaybeCompressedWrite::Compressed(w) => w.write(buf),
            MaybeCompressedWrite::Uncompressed(w) => w.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            MaybeCompressedWrite::Compressed(w) => w.flush(),
            MaybeCompressedWrite::Uncompressed(w) => w.flush(),
        }
    }
}

impl LocalHashedFileOut {
    /// Creates a temporary file to be persisted under a given path.
    ///
    /// The output is compressed with Zlib if `compress` is `true`.
    fn create(base_path: PathBuf, compress: bool) -> Result<Self, Error> {
        let tempfile = NamedTempFile::new()?;
        let (file, temp_path) = tempfile.into_parts();
        let writer = if compress {
            MaybeCompressedWrite::Compressed(
                ZlibEncoder::new(file, Compression::default()),
            )
        } else {
            MaybeCompressedWrite::Uncompressed(file)
        };
        Ok(LocalHashedFileOut {
            temp_path,
            writer,
            base_path,
            context: ring::digest::Context::new(&ring::digest::SHA256),
        })
    }
}

impl Write for LocalHashedFileOut {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.context.update(buf);
        self.writer.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }
}

impl HashedFileOut for LocalHashedFileOut {
    fn persist(mut self, extension: impl AsRef<str>) -> Result<String, Error> {
        self.writer.flush()?;
        self.writer.finish()?;
        if !self.base_path.exists() {
            std::fs::create_dir_all(&self.base_path)?;
        }
        let hash = self.context.finish();
        let hash = base64_engine.encode(&hash);
        let path = self.base_path
            .join(&hash)
            .with_extension(extension.as_ref());
        self.temp_path.persist(path)?;
        Ok(hash)
    }
}

/// Readable file in the local file system.
pub struct LocalHashedFileIn {
    reader: MaybeCompressedRead<File>,
    path: PathBuf,
    // Context to calculate an SHA-256 digest.
    context: ring::digest::Context,
}

// Maybe compressed `Read`.
enum MaybeCompressedRead<R>
where
    R: std::io::Read,
{
    Compressed(ZlibDecoder<R>),
    Uncompressed(R),
}

impl<R> Read for MaybeCompressedRead<R>
where
    R: std::io::Read,
{
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            MaybeCompressedRead::Compressed(r) => r.read(buf),
            MaybeCompressedRead::Uncompressed(r) => r.read(buf),
        }
    }
}

impl LocalHashedFileIn {
    /// Opens a file whose name is the hash of its contents.
    ///
    /// The file must be compressed with `zlib` if `comparessed` is `true`.
    fn open(path: PathBuf, compressed: bool) -> Result<Self, Error> {
        let file = std::fs::File::open(&path)?;
        let reader = if compressed {
            MaybeCompressedRead::Compressed(ZlibDecoder::new(file))
        } else {
            MaybeCompressedRead::Uncompressed(file)
        };
        Ok(LocalHashedFileIn {
            reader,
            path,
            context: ring::digest::Context::new(&ring::digest::SHA256),
        })
    }
}

impl Read for LocalHashedFileIn {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.reader.read(buf)?;
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
