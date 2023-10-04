//! Asynchronous file system.

use async_trait::async_trait;
use base64::engine::{
    Engine,
    general_purpose::URL_SAFE_NO_PAD as url_safe_base_64,
};
use core::mem::{MaybeUninit, transmute};
use core::pin::Pin;
use core::task::Poll;
use flate2::{Decompress, FlushDecompress};
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
        compressed: bool,
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
        compressed: bool,
    ) -> Result<Self::HashedFileIn, Error> {
        LocalHashedFileIn::open(
            self.base_path.join(path.into()),
            compressed,
        ).await
    }
}

pin_project! {
    /// Local file whose name contents can be verified with the hash.
    ///
    /// File name is supposed to be a Base64 encoded URL-safe SHA256 digest of
    /// the contents plus an extension.
    #[must_use = "futures do nothing unless you `.await` or poll them"]
    pub struct LocalHashedFileIn {
        #[pin]
        reader: MaybeCompressedRead<File>,
        hash: String,
        digest: ring::digest::Context,
    }
}

impl LocalHashedFileIn {
    async fn open(path: PathBuf, compressed: bool) -> Result<Self, Error> {
        let hash = path.file_stem()
            .ok_or(Error::InvalidArgs(format!(
                "file name must be hash: {}",
                path.display(),
            )))?
            .to_string_lossy() // should not matter as Base64 is expected
            .to_string();
        let file = File::open(&path).await?;
        let reader = if compressed {
            MaybeCompressedRead::Compressed {
                decoder: AsyncZlibDecoder::new(file),
            }
        } else {
            MaybeCompressedRead::Uncompressed {
                reader: file,
            }
        };
        Ok(Self {
            reader,
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
        match this.reader.poll_read(cx, buf) {
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

const INPUT_BUFFER_SIZE: usize = 4096;

pin_project! {
    /// Zlib decoder that reads bytes from [`AsyncRead`](https://docs.rs/tokio/1.32.0/tokio/io/trait.AsyncRead.html).
    pub struct AsyncZlibDecoder<R> {
        #[pin]
        reader: R,
        reader_finished: bool,
        decoder: Decompress,
        decoder_finished: bool,
        input_buf: [MaybeUninit<u8>; INPUT_BUFFER_SIZE],
        input_pos: usize,
    }
}

impl<R> AsyncZlibDecoder<R> {
    fn new(reader: R) -> Self {
        Self {
            reader,
            reader_finished: false,
            decoder: Decompress::new(true),
            decoder_finished: false,
            input_buf: unsafe { MaybeUninit::uninit().assume_init() },
            input_pos: 0,
        }
    }
}

impl<R> AsyncRead for AsyncZlibDecoder<R>
where
    R: AsyncRead,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        macro_rules! assume_advance {
            ($buf:ident, $amount:expr) => {
                unsafe { $buf.assume_init($buf.filled().len() + $amount); }
                buf.advance($amount);
            };
        }

        let mut this = self.project();
        let initial_len = buf.filled().len();
        let mut input_buf = ReadBuf::uninit(this.input_buf);
        unsafe { input_buf.assume_init(*this.input_pos); }
        input_buf.set_filled(*this.input_pos);
        let mut had_buf_error = false;
        loop {
            if *this.input_pos < input_buf.filled().len()
                && buf.remaining() > 0
            {
                // decompresses the remaining input unless `buf` is full
                let last_total_in = this.decoder.total_in();
                let last_total_out = this.decoder.total_out();
                let input = &input_buf.filled()[*this.input_pos..];
                match this.decoder.decompress(
                    input,
                    unsafe { transmute(buf.unfilled_mut()) },
                    if *this.reader_finished {
                        FlushDecompress::Finish
                    } else {
                        FlushDecompress::None
                    },
                ) {
                    Ok(flate2::Status::Ok) => {
                        let num_written =
                            this.decoder.total_out() - last_total_out;
                        assume_advance!(buf, num_written as usize);
                        let num_read = this.decoder.total_in() - last_total_in;
                        *this.input_pos += num_read as usize;
                        if *this.input_pos == input_buf.filled().len() {
                            input_buf.clear();
                            *this.input_pos = 0;
                        }
                        had_buf_error = false;
                    },
                    Ok(flate2::Status::BufError) => {
                        if had_buf_error {
                            return Poll::Ready(Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                Error::InvalidContext(format!(
                                    "got persisted decoder buffer error",
                                )),
                            )));
                        }
                        had_buf_error = true;
                    },
                    Ok(flate2::Status::StreamEnd) => {
                        *this.decoder_finished = true;
                        let num_written =
                            this.decoder.total_out() - last_total_out;
                        assume_advance!(buf, num_written as usize);
                        let num_read = this.decoder.total_in() - last_total_in;
                        *this.input_pos += num_read as usize;
                        if *this.input_pos == input_buf.filled().len() {
                            input_buf.clear();
                            *this.input_pos = 0;
                        } else {
                            return Poll::Ready(Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                Error::InvalidData(format!(
                                    "extra bytes after compressed block",
                                )),
                            )));
                        }
                        had_buf_error = false;
                    },
                    Err(err) => {
                        return Poll::Ready(Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            err,
                        )));
                    },
                };
            }
            if !*this.reader_finished && input_buf.remaining() > 0 {
                // reads more bytes from the reader
                // unless the reader has finished, or input_buf is full
                let last_len = input_buf.filled().len();
                match this.reader
                    .as_mut()
                    .poll_read(cx, &mut input_buf)
                {
                    Poll::Ready(Ok(_)) => {
                        if input_buf.filled().len() == last_len {
                            *this.reader_finished = true;
                        } else if *this.decoder_finished {
                            return Poll::Ready(Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                Error::InvalidData(format!(
                                    "extra bytes after compressed block",
                                )),
                            )));
                        }
                    },
                    Poll::Pending => {
                        if buf.filled().len() > initial_len {
                            return Poll::Ready(Ok(()));
                        } else {
                            return Poll::Pending;
                        }
                    },
                    Poll::Ready(err) => return Poll::Ready(err),
                }
            }
            if *this.decoder_finished && *this.reader_finished {
                return Poll::Ready(Ok(()));
            }
        }
    }
}

pin_project! {
    /// Maybe compressed [`AsyncRead`](https://docs.rs/tokio/1.32.0/tokio/io/trait.AsyncRead.html).
    #[project = MaybeCompressedReadProj]
    pub enum MaybeCompressedRead<R> {
        /// Zlib-compressed data.
        Compressed {
            #[pin]
            decoder: AsyncZlibDecoder<R>,
        },
        /// Uncompressed data.
        Uncompressed{
            #[pin]
            reader: R,
        },
    }
}

impl<R> AsyncRead for MaybeCompressedRead<R>
where
    R: AsyncRead,
{
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.project() {
            MaybeCompressedReadProj::Compressed { decoder } =>
                decoder.poll_read(cx, buf),
            MaybeCompressedReadProj::Uncompressed { reader } =>
                reader.poll_read(cx, buf),
        }
    }
}
