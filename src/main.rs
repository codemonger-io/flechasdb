use anyhow::Error;
use rand::Rng;

use flechasdb::kmeans::{ Codebook, cluster };
use flechasdb::linalg::{ dot, norm2, scale_in, subtract_in };
use flechasdb::vector::{ BlockVectorSet, VectorSet, divide_vector_set };

fn main() -> Result<(), Error> {
    const N: usize = 5000; // number of vectors
    const M: usize = 1024; // vector size
    const D: usize = 8; // number of divisions
    const C: usize = 25; // number of clusters
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
    let divided = divide_vector_set(&vs, D.try_into()?)?;
    println!("prepared data in: {} us", time.elapsed().as_micros());
    // builds codebooks
    let time = std::time::Instant::now();
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
        }
        codebooks.push(codebook);
    }
    println!("built codebooks in: {} us", time.elapsed().as_micros());
    // builds a distance table for a query vector
    let time = std::time::Instant::now();
    let mut v = vec![0.0f32; M];
    rng.fill(&mut v[..]);
    let v_norm2 = norm2(&v);
    scale_in(&mut v, 1.0 / v_norm2);
    let mut cosine_table: Vec<f32> = Vec::with_capacity(D * C);
    for i in 0..D {
        let from = i * MD;
        let to = from + MD;
        let subv = &v[from..to];
        for j in 0..C {
            let centroid = codebooks[i].centroids.get(j);
            cosine_table.push(dot(subv, centroid));
        }
    }
    let mut distance_table: Vec<f32> = Vec::with_capacity(D * C);
    let mut vector_buf = vec![0.0f32; MD];
    for i in 0..D {
        let from = i * MD;
        let to = from + MD;
        let subv = &v[from..to];
        for j in 0..C {
            let d = &mut vector_buf[..];
            d.copy_from_slice(subv);
            let centroid = codebooks[i].centroids.get(j);
            subtract_in(d, centroid);
            distance_table.push(dot(d, d));
        }
    }
    println!("calculated distance table in: {} us", time.elapsed().as_micros());
    // calculates distances between the query vector and data vectors
    let time = std::time::Instant::now();
    let mut cosines: Vec<(usize, f32)> = Vec::with_capacity(N);
    let mut distances: Vec<(usize, f32)> = Vec::with_capacity(N);
    let mut cos_approxes: Vec<(usize, f32)> = Vec::with_capacity(N);
    let mut dis_approxes: Vec<(usize, f32)> = Vec::with_capacity(N);
    let mut vector_buf = vec![0.0f32; M];
    for i in 0..N {
        let cosine = 1.0 - dot(&v, vs.get(i));
        let d = &mut vector_buf[..];
        d.copy_from_slice(&v);
        subtract_in(d, vs.get(i));
        let distance = dot(d, d);
        let mut cos_approx = 1.0f32;
        for j in 0..D {
            let index = codebooks[j].indices[i];
            cos_approx -= cosine_table[j * C + index];
        }
        let mut dis_approx = 0.0f32;
        for j in 0..D {
            let index = codebooks[j].indices[i];
            dis_approx += distance_table[j * C + index];
        }
        cosines.push((i, cosine));
        distances.push((i, distance));
        cos_approxes.push((i, cos_approx));
        dis_approxes.push((i, dis_approx));
    }
    distances.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    cosines.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    cos_approxes.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    dis_approxes.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    println!("calculated distances in: {} us", time.elapsed().as_micros());
    println!(
        "{}-nearest neighbors:\ncosine:\n{:?}\napprox:\n{:?}\ndistance:\n{:?}\napprox:\n{:?}",
        K,
        &cosines[0..K],
        &cos_approxes[0..K],
        &distances[0..K],
        &dis_approxes[0..K],
    );
    Ok(())
}
