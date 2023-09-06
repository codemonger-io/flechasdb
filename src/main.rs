use anyhow::Error;
use rand::Rng;

use flechasdb::kmeans::{ Codebook, cluster };
use flechasdb::linalg::{ dot, norm2, scale_in, subtract_in };
use flechasdb::partitions::{ Partitioning };
use flechasdb::vector::{ BlockVectorSet, VectorSet, divide_vector_set };

fn main() -> Result<(), Error> {
    const N: usize = 5000; // number of vectors
    const M: usize = 1024; // vector size
    const D: usize = 8; // number of divisions
    const P: usize = 10; // number of partitions
    const C: usize = 25; // number of clusters for PQ
    const MD: usize = M / D; // subvector size
    const K: usize = 10; // K-nearest neighbors
    let use_rkm = false;
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
    println!("prepared data in: {} us", time.elapsed().as_micros());
    // partitions all the data
    let time = std::time::Instant::now();
    let partitions = vs.partition(P.try_into().unwrap())?;
    println!("partitioned data in: {} us", time.elapsed().as_micros());
    // builds codebooks for residues
    let time = std::time::Instant::now();
    let divided = divide_vector_set(&partitions.residues, D.try_into()?)?;
    let mut codebooks: Vec<Codebook<f32>> = Vec::with_capacity(D);
    for subvs in &divided {
        assert_eq!(subvs.vector_size(), MD);
        println!("clustering");
        let codebook: Codebook<f32> = if use_rkm {
            // we can wrap the internal slice of subvs with ArrayView,
            // though, I want to keep `vector` module away from `ndarray.
            let mut block: Vec<f32> = Vec::with_capacity(N * MD);
            for i in 0..N {
                block.extend_from_slice(subvs.get(i));
            }
            let (centroids, indices) = rkm::kmeans_lloyd(
                &ndarray::ArrayView::from_shape((N, MD), &block).unwrap(),
                C.try_into().unwrap(),
            );
            Codebook {
                centroids: BlockVectorSet::chunk(
                    centroids.into_raw_vec(),
                    MD.try_into().unwrap(),
                ).unwrap(),
                indices,
            }
        } else {
            cluster(subvs, C.try_into()?)?
        };
        // intra cluster variance
        /*
        let mut vector_buf = vec![0.0f32; MD];
        for i in 0..C {
            let mut var = 0.0f32;
            let mut count: usize = 0;
            let centroid = codebook.centroids.get(i);
            let d = &mut vector_buf[..];
            for (j, _) in codebook.indices
                    .iter()
                    .enumerate()
                    .filter(|(_, &ci)| ci == i)
            {
                d.copy_from_slice(subvs.get(j));
                subtract_in(d, centroid);
                var += dot(d, d);
                count += 1;
            }
            var = if count > 1 { var / (count - 1) as f32 } else { var };
            println!("cluster {} variance: {}", i, var);
        } */
        codebooks.push(codebook);
    }
    println!("built codebooks in: {} us", time.elapsed().as_micros());
    // creates a random query vector
    let mut qv = vec![0.0f32; M];
    rng.fill(&mut qv[..]);
    let qv_norm2 = norm2(&qv);
    scale_in(&mut qv, 1.0 / qv_norm2);
    // calculates distances from partition centroids
    let time = std::time::Instant::now();
    let mut distances: Vec<f32> = Vec::with_capacity(P);
    let mut vector_buf = vec![0.0f32; M];
    for i in 0..P {
        let centroid = partitions.codebook.centroids.get(i);
        let d = &mut vector_buf[..];
        d.copy_from_slice(&qv[..]);
        subtract_in(d, centroid);
        distances.push(norm2(d));
    }
    println!(
        "calculated distances from partition centroids in: {} us\n{:?}",
        time.elapsed().as_micros(),
        {
            let mut indexed: Vec<_> = distances.iter().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
            indexed
        },
    );
    // calculates approximate distances through partitioned PQ
    let time = std::time::Instant::now();
    let mut approxes: Vec<(usize, f32)> = Vec::with_capacity(N);
    let mut vector_buf = vec![0.0f32; M];
    for i in 0..P {
        let time = std::time::Instant::now();
        let mut distance_table = Vec::with_capacity(D * C);
        let centroid = partitions.codebook.centroids.get(i);
        let localized = &mut vector_buf[..];
        localized.copy_from_slice(&qv[..]);
        subtract_in(localized, centroid);
        let mut vector_buf = vec![0.0f32; MD];
        for j in 0..D {
            let from = j * MD;
            let to = from + MD;
            let subqv = &localized[from..to];
            for k in 0..C {
                let d = &mut vector_buf[..];
                d.copy_from_slice(subqv);
                let centroid = codebooks[j].centroids.get(k);
                subtract_in(d, centroid);
                distance_table.push(dot(d, d));
            }
        }
        for (j, _) in partitions.codebook.indices
            .iter()
            .enumerate()
            .filter(|(_, &pi)| pi == i)
        {
            let mut distance = 0.0f32;
            for k in 0..D {
                let centroid_index = codebooks[k].indices[j];
                distance += distance_table[k * C + centroid_index];
            }
            approxes.push((j, distance));
        }
        println!(
            "approximated distances in partition {} in {} us",
            i,
            time.elapsed().as_micros(),
        );
    }
    println!("approximated distances in {} us", time.elapsed().as_micros());
    // calculates the true distances
    let time = std::time::Instant::now();
    let mut distances: Vec<(usize, f32)> = Vec::with_capacity(N);
    for (i, mut v) in partitions.all_vectors().enumerate() {
        let d = &mut v[..];
        subtract_in(d, &qv);
        let distance = dot(d, d);
        distances.push((i, distance));
    }
    println!("calculated distances in {} us", time.elapsed().as_micros());
    // reports the K nearest neighbors
    approxes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!(
        "k-NN results:\ndistance:\n{:?}\napproximation:\n{:?}",
        &distances[0..K],
        &approxes[0..K],
    );
    Ok(())
}
