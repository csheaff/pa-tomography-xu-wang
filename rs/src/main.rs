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


fn get_signals(tar_info: &ArrayView1<f64>, xd:  &Array1<f64>, t: &Array1<f64>, z_targ: f64) -> Array3<f64> {

//    n_det = xd.len();
    let sigs = Array3::<f64>::zeros((t.len(), 91, 91));

    sigs
}



fn main() {
    //    complex2real()

    let z_targ = 15.0;
    let tar_info: Array2<f64> = 1e-3 * array![[18.0, 0.0, z_targ, 1.5],[-18.0, 0.0, z_targ, 1.5],[9.0, 0.0, z_targ, 1.5],[-9.0, 0.0, z_targ, 1.5],[0.0, 0.0, z_targ, 1.5],[0.0, 12.0, z_targ, 4.0],[0.0, -12.0, z_targ, 4.0]];
    let n_targ = tar_info.shape()[0];
    
    let aper_len = 60e-3;
    let det_pitch = (2.0 / 3.0) * 1e-3;
    let xd = Array::range(-aper_len / 2.0, aper_len / 2.0 + det_pitch, det_pitch);
    let n_det = xd.len();

    let fs = 20e6;
    let ts = 1.0 / fs;
    let t = Array::range(0.0, 65e-6 + ts, ts);

    let mut sigs = Array3::<f64>::zeros((t.len(), n_det, n_det));

    for n in 0..n_targ {
	println!("Generating recorded signals arising from target {} of {}", n + 1, n_targ);
	let ti_slice = tar_info.slice(s![n, ..]);
	sigs = sigs + get_signals(&ti_slice, &xd, &t, z_targ * 1e-3);

    }
//    println!("{:?}", xd)
}
