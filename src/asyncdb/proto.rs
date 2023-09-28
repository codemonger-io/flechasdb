//! Asynchronous utilities for Protocol Buffers.

use protobuf::Message;
use tokio::io::{AsyncRead, AsyncReadExt};

use crate::error::Error;

/// Reads a message from a given
/// [`AsyncRead`](https://docs.rs/tokio/1.32.0/tokio/io/trait.AsyncRead.html).
pub async fn read_message<M, R>(r: &mut R) -> Result<M, Error>
where
    M: Message,
    R: AsyncRead + Unpin + ?Sized,
{
    let mut buf: Vec<u8> = Vec::with_capacity(1024 * 1024);
    r.read_to_end(&mut buf).await?;
    let m = M::parse_from_bytes(&buf)?;
    Ok(m)
}
