# flechasdb

A lightweight vector database in your hands.

`flechasdb` package is the core library of the [FlechasDB system](#flechasdb-system) written in [Rust](https://www.rust-lang.org).

**Table of Contents**

<!-- TOC -->

- [flechasdb](#flechasdb)
    - [FlechasDB system](#flechasdb-system)
        - [Core features and progress](#core-features-and-progress)
    - [Installing flechasdb](#installing-flechasdb)
    - [Using flechasdb](#using-flechasdb)
        - [Building a vector database](#building-a-vector-database)
        - [Loading and querying vector database](#loading-and-querying-vector-database)
        - [Loading and querying vector database async](#loading-and-querying-vector-database-async)
    - [Benchmark](#benchmark)
    - [API documentation](#api-documentation)
    - [Algorithms and structures](#algorithms-and-structures)
        - [IndexIVFPQ](#indexivfpq)
        - [k-means++](#k-means)
        - [Database structure](#database-structure)
    - [Development](#development)
        - [Building the library](#building-the-library)
        - [Generating documentation](#generating-documentation)
    - [Similar projects](#similar-projects)

<!-- /TOC -->

## FlechasDB system

The FlechasDB system is aiming to be a [vector database](https://www.pinecone.io/learn/vector-database/) that perfectly fits in [serverless](https://en.wikipedia.org/wiki/Serverless_computing) environments.
The creed of the FlechasDB system is simple; it requires **no dedicated server continously running**.

### Core features and progress

- [x] Build a vector database from a set of vectors
    - [x] Attach attributes to individual vectors
        - [x] String
        - [x] Number
- [ ] Save a vector database to storage
    - [x] Sync
        - [x] Local file system
        - [x] [Amazon S3](https://aws.amazon.com/s3/) (\*)
    - [ ] Async
        - [ ] Local file system
        - [ ] [Amazon S3](https://aws.amazon.com/s3/)
    - [x] f32
    - [ ] f64
- [x] Load a vector database from storage
    - [x] Sync
        - [x] Local file system
        - [x] [Amazon S3](https://aws.amazon.com/s3/) (\*)
    - [x] Async
        - [x] Local file system
        - [x] [Amazon S3](https://aws.amazon.com/s3/) (\*)
    - [x] f32
    - [ ] f64
- [x] Query vector
    - [x] Sync
        - [x] Get attributes attached to individual vectors
            - [x] String
            - [x] Number
    - [x] Async
        - [x] Get attributes attached to individual vectors
            - [x] String
            - [x] Number
- [ ] Update database
- [ ] Flat database

\*: provided by another package [`flechasdb-s3`](https://github.com/codemonger-io/flechasdb-s3).

## Installing flechasdb

There is no crate published yet.
Please add the following line to your `Cargo.toml` file:

```toml
[dependencies]
flechasdb = { git = "https://github.com/codemonger-io/flechasdb.git" }
```

## Using flechasdb

### Building a vector database

Here is an exmple of building a vector database from randomly generated vectors.

```rs
use rand::Rng;

use flechasdb::db::build::{
    DatabaseBuilder,
    proto::serialize_database,
};
use flechasdb::io::LocalFileSystem;
use flechasdb::vector::BlockVectorSet;

fn main() {
    const M: usize = 100000; // number of vectors
    const N: usize = 1536; // vector size
    const D: usize = 12; // number of subvector divisions
    const P: usize = 100; // number of partitions
    const C: usize = 256; // number of clusters for product quantization
    let time = std::time::Instant::now();
    let mut data: Vec<f32> = Vec::with_capacity(M * N);
    unsafe { data.set_len(M * N); }
    let mut rng = rand::thread_rng();
    rng.fill(&mut data[..]);
    let vs = BlockVectorSet::chunk(data, N.try_into().unwrap()).unwrap();
    println!("prepared data in {} s", time.elapsed().as_secs_f32());
    let time = std::time::Instant::now();
    let mut db = DatabaseBuilder::new(vs)
        .with_partitions(P.try_into().unwrap())
        .with_divisions(D.try_into().unwrap())
        .with_clusters(C.try_into().unwrap())
        .build()
        .unwrap();
    println!("built database in {} s", time.elapsed().as_secs_f32());
    for i in 0..M {
        db.set_attribute_at(i, ("datum_id", i as u64)).unwrap();
    }
    let time = std::time::Instant::now();
    serialize_database(&db, &mut LocalFileSystem::new("testdb")).unwrap();
    println!("serialized database in {} s", time.elapsed().as_secs_f32());
}
```

You can find the complete example in [`examples/build-random`](./examples/build-random/) folder.

FYI: It took a while on my machine (Apple M1 Pro, 32GB RAM, 1TB SSD).
```
prepared data in 0.9123601 s
built database in 906.51526 s
serialized database in 0.14329213 s
```

### Loading and querying vector database

Here is an example of loading a vector database and querying a randomly generated vector for k-nearest neighbors (k-NN).

```rs
use rand::Rng;
use std::env::args;
use std::path::Path;

use flechasdb::db::stored::{Database, LoadDatabase};
use flechasdb::io::LocalFileSystem;

fn main() {
    const K: usize = 10; // k-nearest neighbors
    const NPROBE: usize = 5; // number of partitions to query
    let time = std::time::Instant::now();
    let db_path = args().nth(1).expect("no db path given");
    let db_path = Path::new(&db_path);
    let db = Database::<f32, _>::load_database(
        LocalFileSystem::new(db_path.parent().unwrap()),
        db_path.file_name().unwrap().to_str().unwrap(),
    ).unwrap();
    println!("loaded database in {} s", time.elapsed().as_secs_f32());
    let mut qv: Vec<f32> = Vec::with_capacity(db.vector_size());
    unsafe { qv.set_len(db.vector_size()); }
    let mut rng = rand::thread_rng();
    rng.fill(&mut qv[..]);
    for r in 0..2 { // second round should run faster
        let time = std::time::Instant::now();
        let results = db.query(
            &qv,
            K.try_into().unwrap(),
            NPROBE.try_into().unwrap(),
        ).unwrap();
        println!("[{}] queried k-NN in {} s", r, time.elapsed().as_secs_f32());
        let time = std::time::Instant::now();
        for (i, result) in results.into_iter().enumerate() {
            // getting attributes will incur additional disk reads
            let attr = result.get_attribute("datum_id").unwrap();
            println!(
                "\t{}: partition={}, approx. distance²={}, datum_id={:?}",
                i,
                result.partition_index,
                result.squared_distance,
                attr,
            );
        }
        println!(
            "[{}] printed results in {} s",
            r,
            time.elapsed().as_secs_f32(),
        );
    }
}
```

You can find the complete example in [`examples/query-sync`](./examples/query-sync) folder.

FYI: outputs on my machine (Apple M1 Pro, 32GB RAM, 1TB SSD):
```
loaded database in 0.000142083 s
[0] queried k-NN in 0.0078015 s
	0: partition=95, approx. distance²=126.23533, datum_id=Some(Uint64(90884))
	1: partition=29, approx. distance²=127.76597, datum_id=Some(Uint64(30864))
	2: partition=95, approx. distance²=127.80611, datum_id=Some(Uint64(75236))
	3: partition=56, approx. distance²=127.808174, datum_id=Some(Uint64(27890))
	4: partition=25, approx. distance²=127.85459, datum_id=Some(Uint64(16417))
	5: partition=95, approx. distance²=127.977425, datum_id=Some(Uint64(70910))
	6: partition=25, approx. distance²=128.06209, datum_id=Some(Uint64(3237))
	7: partition=95, approx. distance²=128.22603, datum_id=Some(Uint64(41942))
	8: partition=79, approx. distance²=128.26906, datum_id=Some(Uint64(89799))
	9: partition=25, approx. distance²=128.27995, datum_id=Some(Uint64(6593))
[0] printed results in 0.003392833 s
[1] queried k-NN in 0.001475625 s
	0: partition=95, approx. distance²=126.23533, datum_id=Some(Uint64(90884))
	1: partition=29, approx. distance²=127.76597, datum_id=Some(Uint64(30864))
	2: partition=95, approx. distance²=127.80611, datum_id=Some(Uint64(75236))
	3: partition=56, approx. distance²=127.808174, datum_id=Some(Uint64(27890))
	4: partition=25, approx. distance²=127.85459, datum_id=Some(Uint64(16417))
	5: partition=95, approx. distance²=127.977425, datum_id=Some(Uint64(70910))
	6: partition=25, approx. distance²=128.06209, datum_id=Some(Uint64(3237))
	7: partition=95, approx. distance²=128.22603, datum_id=Some(Uint64(41942))
	8: partition=79, approx. distance²=128.26906, datum_id=Some(Uint64(89799))
	9: partition=25, approx. distance²=128.27995, datum_id=Some(Uint64(6593))
[1] printed results in 0.0000215 s
```

### Loading and querying vector database (async)

Here is an example of asynchronously loading a vector database and querying a randomly generated vector for k-NN.

```rs
use rand::Rng;
use std::env::args;
use std::path::Path;

use flechasdb::asyncdb::io::LocalFileSystem;
use flechasdb::asyncdb::stored::{Database, LoadDatabase};

#[tokio::main]
async fn main() {
    const K: usize = 10; // k-nearest neighbors
    const NPROBE: usize = 5; // number of partitions to search
    let time = std::time::Instant::now();
    let db_path = args().nth(1).expect("missing db path");
    let db_path = Path::new(&db_path);
    let db = Database::<f32, _>::load_database(
        LocalFileSystem::new(db_path.parent().unwrap()),
        db_path.file_name().unwrap().to_str().unwrap(),
    ).await.unwrap();
    println!("loaded database in {} s", time.elapsed().as_secs_f32());
    let mut qv = Vec::with_capacity(db.vector_size());
    unsafe { qv.set_len(db.vector_size()); }
    let mut rng = rand::thread_rng();
    rng.fill(&mut qv[..]);
    for r in 0..2 { // second round should run faster
        let time = std::time::Instant::now();
        let results = db.query(
            &qv,
            K.try_into().unwrap(),
            NPROBE.try_into().unwrap(),
        ).await.unwrap();
        println!("[{}] queried k-NN in {} s", r, time.elapsed().as_secs_f32());
        let time = std::time::Instant::now();
        for (i, result) in results.into_iter().enumerate() {
            // getting attributes will incur additional disk reads
            let attr = result.get_attribute("datum_id").await.unwrap();
            println!(
                "\t{}: partition={}, approx. distance²={}, datum_id={:?}",
                i,
                result.partition_index,
                result.squared_distance,
                attr,
            );
        }
        println!(
            "[{}] printed results in {} s",
            r,
            time.elapsed().as_secs_f32(),
        );
    }
}
```

The complete example is in [`examples/query-async`](./examples/query-async) folder.

FYI: outputs on my machine (Apple M1 Pro, 32GB RAM, 1TB SSD):
```
loaded database in 0.000170959 s
[0] queried k-NN in 0.008041208 s
	0: partition=67, approx. distance²=128.50703, datum_id=Some(Uint64(69632))
	1: partition=9, approx. distance²=129.98079, datum_id=Some(Uint64(73093))
	2: partition=9, approx. distance²=130.10867, datum_id=Some(Uint64(7536))
	3: partition=20, approx. distance²=130.29523, datum_id=Some(Uint64(67750))
	4: partition=67, approx. distance²=130.71976, datum_id=Some(Uint64(77054))
	5: partition=9, approx. distance²=130.80556, datum_id=Some(Uint64(93180))
	6: partition=9, approx. distance²=130.90681, datum_id=Some(Uint64(22473))
	7: partition=9, approx. distance²=130.94006, datum_id=Some(Uint64(40167))
	8: partition=67, approx. distance²=130.9795, datum_id=Some(Uint64(8590))
	9: partition=9, approx. distance²=131.03018, datum_id=Some(Uint64(53138))
[0] printed results in 0.00194175 s
[1] queried k-NN in 0.000789417 s
	0: partition=67, approx. distance²=128.50703, datum_id=Some(Uint64(69632))
	1: partition=9, approx. distance²=129.98079, datum_id=Some(Uint64(73093))
	2: partition=9, approx. distance²=130.10867, datum_id=Some(Uint64(7536))
	3: partition=20, approx. distance²=130.29523, datum_id=Some(Uint64(67750))
	4: partition=67, approx. distance²=130.71976, datum_id=Some(Uint64(77054))
	5: partition=9, approx. distance²=130.80556, datum_id=Some(Uint64(93180))
	6: partition=9, approx. distance²=130.90681, datum_id=Some(Uint64(22473))
	7: partition=9, approx. distance²=130.94006, datum_id=Some(Uint64(40167))
	8: partition=67, approx. distance²=130.9795, datum_id=Some(Uint64(8590))
	9: partition=9, approx. distance²=131.03018, datum_id=Some(Uint64(53138))
[1] printed results in 0.000011084 s
```

## Benchmark

There is a [benchmark](https://github.com/codemonger-io/flechasdb-benchmark) on more realistic data.

## API documentation

https://codemonger-io.github.io/flechasdb/api/flechasdb/

## Algorithms and structures

### IndexIVFPQ

`flechasdb` implements IndexIVFPQ described in [this article](https://mccormickml.com/2017/10/22/product-quantizer-tutorial-part-2/).

### k-means++

`flechasdb` implements [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) to initialize centroids for näive k-means clustering.

### Database structure

TBD

## Development

### Building the library

```sh
cargo build
```

### Generating documentation

```sh
cargo doc --lib --no-deps --release
```

## Similar projects

- [Pinecone](https://www.pinecone.io)

  Fully managed vector database.

- [LanceDB](https://lancedb.com)

  One of their features is also **serverless**.