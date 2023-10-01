//! The core library of the FlechasDB system.

#![warn(missing_docs)]

pub mod asyncdb;
pub mod db;
pub mod distribution;
pub mod error;
pub mod io;
pub mod kmeans;
pub mod linalg;
pub mod nbest;
pub mod numbers;
pub mod partitions;
pub mod protos;
pub mod slice;
pub mod vector;
