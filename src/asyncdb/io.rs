//! Asynchronous file system.

use async_trait::async_trait;
use base64::engine::{
    Engine,
    general_purpose::URL_SAFE_NO_PAD as url_safe_base_64,
};
use core::pin::Pin;
use core::task::Poll;
use pin_project_lite::pin_project;
use std::path::{Path, PathBuf};
use tokio::fs::File;
use tokio::io::{AsyncRead, ReadBuf};

use crate::error::Error;

/// Asynchronous file system.
#[async_trait]
pub trait FileSystem {
    /// File whose contents can be verified with the hash.
    type HashedFileIn: HashedFileIn;

    /// Opens a file whose contents can be verified with the hash.
    async fn open_hashed_file(
        &self,
        path: impl Into<String> + Send,
    ) -> Result<Self::HashedFileIn, Error>;
}

/// File whose contents can be verified with the hash.
#[async_trait]
pub trait HashedFileIn: AsyncRead + Send + Unpin {
    /// Verifies the file contents.
    ///
    /// Finishes the calculation of the hash and verifies the contents.
    /// You should call this function after the entire file has been read.
    ///
    /// File name is supposed to contain a Base64 encoded URL-safe SHA256
    /// digest of the contents, but it is up to implementation.
    ///
    /// Fails with `Error::VerificationFailure` if the contents cannot be
    /// verified.
    async fn verify(self) -> Result<(), Error>;
}

/// Asynchronous local file system.
pub struct LocalFileSystem {
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

#[async_trait]
impl FileSystem for LocalFileSystem {
    type HashedFileIn = LocalHashedFileIn;

    async fn open_hashed_file(
        &self,
        path: impl Into<String> + Send,
    ) -> Result<Self::HashedFileIn, Error> {
        LocalHashedFileIn::open(self.base_path.join(path.into())).await
    }
}

pin_project! {
    /// Local file whose name contents can be verified with the hash.
    ///
    /// File name is supposed to be a Base64 encoded URL-safe SHA256 digest of
    /// the contents plus an extension.
    #[must_use = "futures do nothing unless you `.await` or poll them"]
    pub struct LocalHashedFileIn {
        file: File,
        hash: String,
        digest: ring::digest::Context,
    }
}

impl LocalHashedFileIn {
    async fn open(path: PathBuf) -> Result<Self, Error> {
        let hash = path.file_stem()
            .ok_or(Error::InvalidArgs(format!(
                "file name must be hash: {}",
                path.display(),
            )))?
            .to_string_lossy() // should not matter as Base64 is expected
            .to_string();
        let file = File::open(&path).await?;
        Ok(Self {
            file,
            hash,
            digest: ring::digest::Context::new(&ring::digest::SHA256),
        })
    }
}

#[async_trait]
impl HashedFileIn for LocalHashedFileIn {
    async fn verify(self) -> Result<(), Error> {
        let digest = self.digest.finish();
        let hash = url_safe_base_64.encode(digest);
        if self.hash == hash {
            Ok(())
        } else {
            Err(Error::VerificationFailure(format!(
                "hash discrepancy: expected {} but got {}",
                self.hash,
                hash,
            )))
        }
    }
}

impl AsyncRead for LocalHashedFileIn {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let this = self.project();
        let last_len = buf.filled().len();
        match Pin::new(this.file).poll_read(cx, buf) {
            Poll::Ready(Ok(())) => {
                if buf.filled().len() != last_len {
                    let buf = &buf.filled()[last_len..];
                    this.digest.update(buf);
                }
                Poll::Ready(Ok(()))
            },
            Poll::Pending => Poll::Pending,
            Poll::Ready(err) => Poll::Ready(err),
        }
    }
}
