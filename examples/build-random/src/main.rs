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
