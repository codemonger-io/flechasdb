//! Asynchronous request for an attribute.

use core::borrow::Borrow;
use core::future::Future;
use core::hash::Hash;
use core::pin::Pin;
use core::task::{Context, Poll};
use pin_project_lite::pin_project;
use uuid::Uuid;

use crate::db::AttributeValue;
use crate::error::Error;

use super::{AttributeValueRef, Database, LoadAttributesLog};

pin_project! {
    /// Asynchronous request for an attribute in a specific partition.
    #[must_use = "futures do nothing unless you `.await` or poll them"]
    pub struct GetAttributeInPartition<'db, 'i, 'k, T, FS, K>
    where
        T: Send,
        FS: Send,
        K: ?Sized,
    {
        db: &'db Database<T, FS>,
        partition_index: usize,
        vector_id: &'i Uuid,
        key: &'k K,
        #[pin]
        load_attributes_log: Option<Pin<Box<
            dyn 'db + Future<Output = Result<(), Error>>,
        >>>,
        #[pin]
        get_attribute_internal: Option<Pin<Box<
            dyn 'db + Future<Output = Result<Option<AttributeValueRef<'db>>, Error>>,
        >>>,
    }
}

impl<'db, 'i, 'k, T, FS, K> GetAttributeInPartition<'db, 'i, 'k, T, FS, K>
where
    T: Send,
    FS: Send,
    K: ?Sized,
{
    /// Creates a new asynchronous request for an attribute in a specific
    /// partition.
    pub(super) fn new(
        db: &'db Database<T, FS>,
        partition_index: usize,
        vector_id: &'i Uuid,
        key: &'k K,
    ) -> Self {
        GetAttributeInPartition {
            db,
            partition_index,
            vector_id,
            key,
            load_attributes_log: None,
            get_attribute_internal: None,
        }
    }
}

impl<'db, 'i, 'k, T, FS, K> Future for GetAttributeInPartition<'db, 'i, 'k, T, FS, K>
where
    T: Send,
    FS: Send,
    String: Borrow<K>,
    K: Hash + Eq + ?Sized,
    Database<T, FS>: LoadAttributesLog<'db>,
    'i: 'db,
    'k: 'db,
{
    type Output = Result<Option<AttributeValue>, Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut this = self.project();
        loop {
            if let Some(future) = this.get_attribute_internal
                .as_mut()
                .as_pin_mut()
            {
                // 3. waits for the attribute value
                match future.poll(cx) {
                    Poll::Ready(Ok(result)) => {
                        let value = result.map(|value| value.clone());
                        return Poll::Ready(Ok(value));
                    },
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                }
            } else if let Some (future) = this.load_attributes_log
                .as_mut()
                .as_pin_mut()
            {
                // 2. requests for the attribute value
                match future.poll(cx) {
                    Poll::Ready(Ok(_)) => {
                        *this.get_attribute_internal = Some(Box::pin(
                            this.db.get_attribute_internal(
                                this.vector_id,
                                this.key,
                            ),
                        ));
                    },
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Err(err)) => return Poll::Ready(Err(err)),
                };
            } else {
                // 1. loads the attributes log
                *this.load_attributes_log = Some(
                    this.db.load_attributes_log(*this.partition_index),
                );
            }
        }
    }
}
