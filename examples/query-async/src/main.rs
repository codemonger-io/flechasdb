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
