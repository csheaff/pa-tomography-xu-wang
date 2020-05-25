extern crate ndarray;
use ndarray::prelude::*;
//use std::f64::consts::*;
//use std::path::PathBuf;
use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;
use std::vec::Vec;

// fn complex2complex() {
//     let n = 128;
//     let mut plan: C2CPlan64 = C2CPlan::aligned(&[n], Sign::Forward, Flag::Measure).unwrap();
//     let mut a = AlignedVec::new(n);
//     let mut b = AlignedVec::new(n);
//     let k0 = 2.0 * PI / n as f64;
//     for i in 0..n {
// 	a[i] = c64::new((k0 * i as f64).cos(), 0.0);
//     }
//     plan.c2c(&mut a, &mut b).unwrap();

//     println!("hello world");

// }


fn complex2real() {

    let n = 128;
    let mut c2r: C2RPlan64 = C2RPlan::aligned(&[n], Flag::Measure).unwrap();
    let mut a = AlignedVec::new(n / 2 + 1);
    let mut b = AlignedVec::new(n);
    for i in 0..(n / 2 + 1) {
	a[i] = c64::new(1.0, 0.0);
    }
    c2r.c2r(&mut a, &mut b).unwrap();

    println!("{:?}", b);

    let c = Array::from(Vec::from(b.as_slice()));

    println!("{:?}", c);
}


fn main() {
    complex2real()
}
