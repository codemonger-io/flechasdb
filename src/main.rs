use anyhow::Error;
use rand::Rng;

use flechasdb::db::{
    Database,
    DatabaseBuilder,
    DatabaseBuilderEvent,
    DatabaseQueryEvent,
};
use flechasdb::db::proto::serialize_database;
use flechasdb::io::LocalFileSystem;
use flechasdb::linalg::{ norm2, scale_in };
use flechasdb::vector::{ BlockVectorSet, VectorSet };

fn main() -> Result<(), Error> {
    const N: usize = 5000; // number of vectors
    const M: usize = 1024; // vector size
    const D: usize = 8; // number of divisions
    const P: usize = 10; // number of partitions
    const C: usize = 25; // number of clusters for product quantization
    const K: usize = 10; // K-nearest neighbors
    const NP: usize = 3; // number of partitions to query
    // prepares the data
    let time = std::time::Instant::now();
    let mut data = vec![0.0f32; N * M];
    let mut rng = rand::thread_rng();
    rng.fill(&mut data[..]);
    let mut vs = BlockVectorSet::chunk(data, M.try_into()?)?;
    for i in 0..N {
        let v = vs.get_mut(i);
        scale_in(v, 1.0 / norm2(v));
    }
    println!("prepared data in {} μs", time.elapsed().as_micros());
    // builds a vector database
    let time = std::time::Instant::now();
    let mut event_time = std::time::Instant::now();
    let db = DatabaseBuilder::new(vs)
        .with_partitions(P.try_into().unwrap())
        .with_divisions(D.try_into().unwrap())
        .with_clusters(C.try_into().unwrap())
        .build(Some(move |event| {
            match event {
                DatabaseBuilderEvent::StartingIdAssignment |
                DatabaseBuilderEvent::StartingPartitioning |
                DatabaseBuilderEvent::StartingSubvectorDivision |
                DatabaseBuilderEvent::StartingQuantization(_) => {
                    event_time = std::time::Instant::now();
                },
                DatabaseBuilderEvent::FinishedIdAssignment => {
                    println!(
                        "assigned vector IDs in {} μs",
                        event_time.elapsed().as_micros(),
                    );
                },
                DatabaseBuilderEvent::FinishedPartitioning => {
                    println!(
                        "partitioned data in {} μs",
                        event_time.elapsed().as_micros(),
                    );
                },
                DatabaseBuilderEvent::FinishedSubvectorDivision => {
                    println!(
                        "divided data in {} μs",
                        event_time.elapsed().as_micros(),
                    );
                },
                DatabaseBuilderEvent::FinishedQuantization(i) => {
                    println!(
                        "quantized division {} in {} μs",
                        i,
                        event_time.elapsed().as_micros(),
                    );
                },
            };
        }))?;
    println!("built database in {} μs", time.elapsed().as_micros());
    // creates a random query vector
    let mut qv = vec![0.0f32; M];
    rng.fill(&mut qv[..]);
    let qv_norm2 = norm2(&qv);
    scale_in(&mut qv, 1.0 / qv_norm2);
    // queries k-NN
    let time = std::time::Instant::now();
    let mut event_time = std::time::Instant::now();
    let results = db.query(
        &qv,
        K.try_into().unwrap(),
        NP.try_into().unwrap(),
        Some(move |event| {
            match event {
                DatabaseQueryEvent::StartingPartitionSelection |
                DatabaseQueryEvent::StartingPartitionQuery(_) |
                DatabaseQueryEvent::StartingResultSelection => {
                    event_time = std::time::Instant::now();
                },
                DatabaseQueryEvent::FinishedPartitionSelection => {
                    println!(
                        "selected partitions in {} μs",
                        event_time.elapsed().as_micros(),
                    );
                },
                DatabaseQueryEvent::FinishedPartitionQuery(i) => {
                    println!(
                        "queried partition {} in {} μs",
                        i,
                        event_time.elapsed().as_micros(),
                    );
                },
                DatabaseQueryEvent::FinishedResultSelection => {
                    println!(
                        "selected results in {} μs",
                        event_time.elapsed().as_micros(),
                    );
                },
            }
        })
    )?;
    println!("queried k-NN in {} μs", time.elapsed().as_micros());
    for (i, result) in results.iter().enumerate() {
        println!("{}: {:?}", i, result);
    }
    // saves the database
    let time = std::time::Instant::now();
    save_database(&db, "testdb")?;
    println!("saved database in {} μs", time.elapsed().as_micros());
    Ok(())
}

fn save_database<VS, P>(
    db: &Database<f32, VS>,
    base_path: P,
) -> Result<(), Error>
where
    VS: VectorSet<f32>,
    P: AsRef<std::path::Path> + core::fmt::Debug,
{
    println!("saving database to {:?}", base_path);
    let mut fs = LocalFileSystem::new(base_path);
    serialize_database(db, &mut fs)?;
    Ok(())
}
