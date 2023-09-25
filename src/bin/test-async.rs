//! Tests asyncdb.

use anyhow::Error;
use rand::Rng;

use flechasdb::asyncdb::io::LocalFileSystem;
use flechasdb::asyncdb::stored::{Database, LoadDatabase, QueryEvent};
use flechasdb::linalg::{norm2, scale_in};

#[tokio::main]
async fn main() -> Result<(), Error> {
    let args = std::env::args().collect::<Vec<_>>();
    let path_segments = args[1].split('/').collect::<Vec<_>>();
    let base_path = path_segments[..path_segments.len() - 1].join("/");
    let db_name = path_segments[path_segments.len() - 1];
    let fs = LocalFileSystem::new(base_path);

    let time = std::time::Instant::now();
    let db = Database::<f32, _>::load_database(fs, db_name).await?;
    println!("loaded database in {:?} μs", time.elapsed().as_micros());

    let mut rng = rand::thread_rng();
    let qv = random_query_vector(&mut rng, db.vector_size());

    const K: usize = 10;
    const NP: usize = 3;
    let time = std::time::Instant::now();
    let event_time = std::time::Instant::now();
    let results = db.query(
        &qv[..],
        K.try_into().unwrap(),
        NP.try_into().unwrap(),
        Some(move |event| match event {
            QueryEvent::StartingLoadingPartitionCentroids =>
                println!(
                    "starting loading partition centroids at {} μs",
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::FinishedLoadingPartitionCentroids =>
                println!(
                    "finished loading partition centroids at {} μs",
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::StartingLoadingCodebooks =>
                println!(
                    "starting loading codebooks at {} μs",
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::FinishedLoadingCodebooks =>
                println!(
                    "finished loading codebooks at {} μs",
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::StartingPartitionSelection =>
                println!(
                    "starting partition selection at {} μs",
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::FinishedPartitionSelection =>
                println!(
                    "finished partition selection at {} μs",
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::StartingLoadingPartition(i) =>
                println!(
                    "starting loading partition {} at {} μs",
                    i,
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::FinishedLoadingPartition(i) =>
                println!(
                    "finished loading partition {} at {} μs",
                    i,
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::StartingPartitionQueryExecution(i) =>
                println!(
                    "starting partition query execution {} at {} μs",
                    i,
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::FinishedPartitionQueryExecution(i) =>
                println!(
                    "finished partition query execution {} at {} μs",
                    i,
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::StartingKNNSelection =>
                println!(
                    "starting KNN selection at {} μs",
                    event_time.elapsed().as_micros(),
                ),
            QueryEvent::FinishedKNNSelection =>
                println!(
                    "finished KNN selection at {} μs",
                    event_time.elapsed().as_micros(),
                ),
        }),
    ).await?;
    println!("queried database in {:?} μs", time.elapsed().as_micros());

    let time = std::time::Instant::now();
    let results: Result<_, Error> = futures::future::try_join_all(
        results.into_iter().map(|result| async move {
            let datum_id = result.get_attribute("datum_id").await?;
            Ok((datum_id, result))
        }),
    ).await;
    let results = results?;
    for (i, (datum_id, result)) in results.into_iter().enumerate() {
        println!(
            "[{}]: datum_id: {:?}, ID: {:?}, distance: {}, partition: {}",
            i,
            datum_id,
            result.vector_id,
            result.squared_distance,
            result.partition_index,
        );
    }
    println!("iterated results in {:?} μs", time.elapsed().as_micros());

    Ok(())
}

fn random_query_vector<R>(rng: &mut R, size: usize) -> Vec<f32>
where
    R: Rng,
{
    let mut v: Vec<f32> = Vec::with_capacity(size);
    unsafe {
        let p = v.as_mut_ptr();
        rng.fill(core::slice::from_raw_parts_mut(p, size));
        v.set_len(size);
    }
    let norm = norm2(&v);
    scale_in(&mut v, 1.0 / norm);
    v
}
