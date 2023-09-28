# flechasdb

A lightweight vector database in your hands.

`flechasdb` package is the core library of the [FlechasDB system](#flechasdb-system) written in [Rust](https://www.rust-lang.org).

## FlechasDB system

The FlechasDB system is aiming to be a [vector database](https://www.pinecone.io/learn/vector-database/) that perfectly fits in [serverless](https://en.wikipedia.org/wiki/Serverless_computing) environments.
The creed of the FlechasDB system is simple; it requires **no dedicated server continously running**.

### Core features and progress

- [x] Build a vector database from a set of vectors
    - [ ] Attach attributes to individual vectors
        - [x] String
        - [ ] Number
- [ ] Save a vector database to storage
    - [ ] Sync
        - [x] Local file system
        - [ ] [Amazon S3](https://aws.amazon.com/s3/)
    - [ ] Async
        - [ ] Local file system
        - [ ] [Amazon S3](https://aws.amazon.com/s3/)
    - [x] f32
    - [ ] f64
- [x] Load a vector database from storage
    - [x] Sync
        - [x] Local file system
        - [x] [Amazon S3](https://aws.amazon.com/s3/) (\*1)
    - [x] Async
        - [x] Local file system
        - [x] [Amazon S3](https://aws.amazon.com/s3/) (\*1)
    - [x] f32
    - [ ] f64
- [x] Query vector
    - [x] Sync
        - [ ] Get attributes attached to individual vectors
            - [x] String
            - [ ] Number
    - [x] Async
        - [ ] Get attributes attached to individual vectors
            - [x] String
            - [ ] Number
- [ ] Update database
- [ ] Flat database

\*1: provided by another package [`flechasdb-s3`](https://github.com/codemonger-io/flechasdb-s3).

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
    println!("prepared data at {} s", time.elapsed().as_secs_f32());
    let db = DatabaseBuilder::new(vs)
        .with_partitions(P.try_into().unwrap())
        .with_divisions(D.try_into().unwrap())
        .with_clusters(C.try_into().unwrap())
        .build()
        .unwrap();
    println!("built database at {} s", time.elapsed().as_secs_f32());
    serialize_database(&db, &mut LocalFileSystem::new("testdb")).unwrap();
    println!("serialized database at {} s", time.elapsed().as_secs_f32());
}
```

You can find the complete example in [`examples/build-random`](./examples/build-random/) folder.

FYI: It took a while on my machine (Apple M1 Pro, 32GB RAM).
```
prepared data in 0.94093055 s
built database in 870.743 s
serialized database in 0.077745415 s
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
            let attr = db.get_attribute_of(&result, "attr").unwrap();
            println!(
                "\t{}: partition={}, approx. distance²={}, attr={:?}",
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

FYI: outputs on my machine (Apple M1 Pro, 32GB RAM):
```
loaded database in 0.000153583 s
[0] queried k-NN in 0.008891375 s
	0: partition=70, approx. distance²=131.25273, attr=None
	1: partition=76, approx. distance²=131.99782, attr=None
	2: partition=63, approx. distance²=132.21599, attr=None
	3: partition=76, approx. distance²=132.30228, attr=None
	4: partition=63, approx. distance²=132.57605, attr=None
	5: partition=65, approx. distance²=132.68034, attr=None
	6: partition=65, approx. distance²=132.7237, attr=None
	7: partition=63, approx. distance²=132.7903, attr=None
	8: partition=63, approx. distance²=132.91724, attr=None
	9: partition=63, approx. distance²=132.9236, attr=None
[0] printed results in 0.00073575 s
[1] queried k-NN in 0.001442917 s
	0: partition=70, approx. distance²=131.25273, attr=None
	1: partition=76, approx. distance²=131.99782, attr=None
	2: partition=63, approx. distance²=132.21599, attr=None
	3: partition=76, approx. distance²=132.30228, attr=None
	4: partition=63, approx. distance²=132.57605, attr=None
	5: partition=65, approx. distance²=132.68034, attr=None
	6: partition=65, approx. distance²=132.7237, attr=None
	7: partition=63, approx. distance²=132.7903, attr=None
	8: partition=63, approx. distance²=132.91724, attr=None
	9: partition=63, approx. distance²=132.9236, attr=None
[1] printed results in 0.000015541 s
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
            let attr = result.get_attribute("attr").await.unwrap();
            println!(
                "\t{}: partition={}, approx. distance²={}, attr={:?}",
                i,
                result.partition_index,
                result.squared_distance,
                attr,
            );
        }
        println!(
            "[{}] printed results at {} s",
            r,
            time.elapsed().as_secs_f32(),
        );
    }
}
```

The complete example is in [`examples/query-async`](./examples/query-async) folder.

FYI: outputs on my machine (Apple M1 Pro, 32GB RAM):
```
loaded database in 0.000205958 s
[0] queried k-NN in 0.008670959 s
	0: partition=3, approx. distance²=130.65294, attr=None
	1: partition=3, approx. distance²=130.75792, attr=None
	2: partition=3, approx. distance²=130.77882, attr=None
	3: partition=15, approx. distance²=130.82741, attr=None
	4: partition=7, approx. distance²=130.92447, attr=None
	5: partition=46, approx. distance²=131.00838, attr=None
	6: partition=46, approx. distance²=131.03413, attr=None
	7: partition=46, approx. distance²=131.08325, attr=None
	8: partition=2, approx. distance²=131.09665, attr=None
	9: partition=3, approx. distance²=131.31482, attr=None
[0] printed results in 0.00116875 s
[1] queried k-NN in 0.0010745 s
	0: partition=3, approx. distance²=130.65294, attr=None
	1: partition=3, approx. distance²=130.75792, attr=None
	2: partition=3, approx. distance²=130.77882, attr=None
	3: partition=15, approx. distance²=130.82741, attr=None
	4: partition=7, approx. distance²=130.92447, attr=None
	5: partition=46, approx. distance²=131.00838, attr=None
	6: partition=46, approx. distance²=131.03413, attr=None
	7: partition=46, approx. distance²=131.08325, attr=None
	8: partition=2, approx. distance²=131.09665, attr=None
	9: partition=3, approx. distance²=131.31482, attr=None
[1] printed results in 0.000012208 s
```

## API documentation

https://codemonger-io.github.io/flechasdb/api/

## Algorithms and structures

### IndexIVFPQ

`flechasdb` implements IndexIVFPQ described in [this article](https://mccormickml.com/2017/10/22/product-quantizer-tutorial-part-2/).

### k-means++

`flechasdb` implements [k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B) to initialize centroids for näive k-means clustering.

### Database structure

## Development

### Building the libraryo

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