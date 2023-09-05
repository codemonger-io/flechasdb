//! Runs benchmarks.

use rand::Rng;

use flechasdb::linalg::{
    dot,
    dot_naive,
    min,
    min_naive,
    scale_in,
    scale_in_naive,
    subtract_in,
    subtract_in_naive,
    sum,
    sum_naive,
};

fn main() {
    benchmark_dot();
    benchmark_min();
    benchmark_scale_in();
    benchmark_subtract_in();
    benchmark_sum();
    benchmark_locality();
}

fn benchmark_dot() {
    const R: usize = 5;
    const N: usize = 10000000;
    let mut xs = vec![0.0f32; N];
    let mut ys = vec![0.0f32; N];
    let mut rng = rand::thread_rng();
    rng.fill(&mut xs[..]);
    rng.fill(&mut ys[..]);
    for _ in 0..R {
        let time = std::time::Instant::now();
        let ans = dot(&xs, &ys);
        let elapsed = time.elapsed();
        println!("dot {} in {} μs", ans, elapsed.as_micros());
        let time = std::time::Instant::now();
        let ans = dot_naive(&xs, &ys);
        let elapsed = time.elapsed();
        println!("näive dot {} in {} μs", ans, elapsed.as_micros());
    }
}

fn benchmark_min() {
    const R: usize = 5;
    const N: usize = 10000000;
    let mut v = vec![0.0f32; N];
    let mut rng = rand::thread_rng();
    rng.fill(&mut v[..]);
    for _ in 0..R {
        let time = std::time::Instant::now();
        let mn = min(&v);
        let elapsed = time.elapsed();
        println!("min {:?} in {} μs", mn, elapsed.as_micros());
        let time = std::time::Instant::now();
        let mn = min_naive(&v);
        let elapsed = time.elapsed();
        println!("näive min {:?} in {} μs", mn, elapsed.as_micros());
    }
}

fn benchmark_scale_in() {
    const R: usize = 5;
    const N: usize = 10000000;
    let mut xs = vec![0.0f32; N];
    let mut rng = rand::thread_rng();
    let a = rng.gen::<f32>();
    rng.fill(&mut xs[..]);
    for _ in 0..R {
        let time = std::time::Instant::now();
        scale_in(&mut xs, a);
        let elapsed = time.elapsed();
        println!("scale in {} μs", elapsed.as_micros());
        let time = std::time::Instant::now();
        scale_in_naive(&mut xs, a);
        let elapsed = time.elapsed();
        println!("näive scale in {} μs", elapsed.as_micros());
    }
}

fn benchmark_subtract_in() {
    const R: usize = 5;
    const N: usize = 10000000;
    let mut xs = vec![0.0f32; N];
    let mut ys = vec![0.0f32; N];
    let mut rng = rand::thread_rng();
    rng.fill(&mut xs[..]);
    rng.fill(&mut ys[..]);
    for _ in 0..R {
        let time = std::time::Instant::now();
        subtract_in(&mut xs, &ys);
        let elapsed = time.elapsed();
        println!("subtract in {} μs", elapsed.as_micros());
        let time = std::time::Instant::now();
        subtract_in_naive(&mut xs, &ys);
        let elapsed = time.elapsed();
        println!("näive subtract in {} μs", elapsed.as_micros());
    }
}

fn benchmark_sum() {
    const R: usize = 5;
    const N: usize = 10000000;
    let mut v = vec![0.0f32; N];
    let mut rng = rand::thread_rng();
    rng.fill(&mut v[..]);
    for _ in 0..R {
        let time = std::time::Instant::now();
        let s = sum(&v);
        let elapsed = time.elapsed();
        println!("sum {} in {} μs", s, elapsed.as_micros());
        let time = std::time::Instant::now();
        let s = sum_naive(&v);
        let elapsed = time.elapsed();
        println!("näive sum {} in {} μs", s, elapsed.as_micros());
    }
}

fn benchmark_locality() {
    const R: usize = 5;
    const N: usize = 10000000;
    const M: usize = 100;
    const T: usize = N / M - 1;
    let mut xs = vec![0.0f32; N];
    let mut ys = vec![0.0f32; N];
    let mut local = vec![0.0f32; M * 2];
    let mut rng = rand::thread_rng();
    rng.fill(&mut xs[..]);
    rng.fill(&mut ys[..]);
    for _ in 0..R {
        let mut a = 0.0f32;
        let time = std::time::Instant::now();
        for t in 0..T {
            let ls = &xs[t*M..(t+1)*M];
            let rs = &xs[(t+1)*M..(t+2)*M];
            a += dot(ls, rs);
        }
        let elapsed = time.elapsed();
        println!("near dot in {} μs (a={})", elapsed.as_micros(), a);
        let mut a = 0.0f32;
        let time = std::time::Instant::now();
        for t in 0..T {
            let ls = &xs[t*M..(t+1)*M];
            let rs = &ys[t*M..(t+1)*M];
            a += dot(ls, rs);
        }
        let elapsed = time.elapsed();
        println!("far dot in {} μs (a={})", elapsed.as_micros(), a);
        let mut a = 0.0f32;
        let time = std::time::Instant::now();
        for t in 0..T {
            local[..M].copy_from_slice(&xs[t*M..(t+1)*M]);
            local[M..].copy_from_slice(&ys[t*M..(t+1)*M]);
            let ls = &local[..M];
            let rs = &local[M..];
            a += dot(ls, rs);
        }
        let elapsed = time.elapsed();
        println!("copy dot in {} μs (a={})", elapsed.as_micros(), a);
    }
}
