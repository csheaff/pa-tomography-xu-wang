extern crate ndarray;
use ndarray::prelude::*;
use ndarray_linalg::*;
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


fn step_fn(x: &Array1<f64>) -> Array1<f64> {
  let y = 0.5 * (x.mapv(f64::signum) + 1.0);
  y 
}


fn get_signals(tar_info: &ArrayView1<f64>, xd:  &Array1<f64>, t: &Array1<f64>, z_targ: f64) -> Array3<f64> {
    // Clay this is all very slow and I don't know why.
    let yd = xd.clone();
    let det_len = 2e-3;
    let n_subdet = 25;
    let n_subdet_perdim = (n_subdet as f64).sqrt() as i32;
    let subdet_pitch = det_len / n_subdet as f64;
    let subdet_ind = Array::range(0.0, n_subdet as f64, 1.0) - (n_subdet as f64 - 1.0) / 2.0;
    let subdet_offset = subdet_pitch * &subdet_ind;

    let fs = 1.0 / (t[1] - t[0]);
    let fc = 4e6;
    let c = 1484.0;
    
    let n_det_x = xd.len();
    let n_det_y = yd.len();
    let mut sigs = Array3::<f64>::zeros((t.len(), n_det_x, n_det_x));
    for xi in 0..n_det_x {
	println!("{ }", xi);
	for yi in 0..n_det_y {
	    let mut pa_sig = Array1::<f64>::zeros(t.len());
	    for m in 0..n_subdet {
		for n in 0..n_subdet {
 		    let det_xyz = array![xd[xi] + subdet_offset[m], yd[yi] + subdet_offset[n], 0.0];
		    let tar_xyz = tar_info.slice(s![..3]);
		    let r = norm::Norm::norm_l2(&(det_xyz - tar_xyz));
		    let tar_rad = tar_info[3];
		    let step_fn_arg = tar_rad - 1.0* (r - c * t).mapv(f64::abs);
		    let step_fn_arg = t.mapv(f64::abs);  // why is mapv so slow??
		    pa_sig = pa_sig + step_fn(&step_fn_arg) * (r - c * t) / (2.0 * r);
		}
	    }
	    let pr = pa_sig / n_subdet as f64;
	    let mut slice = sigs.slice_mut(s![.., xi, yi]);
	    slice.assign(&pr);
	}
    }
    
    
    
//    println!("{:?}", subdet_offset);
    
    sigs
}



fn main() {
    //    complex2real()

    let z_targ = 15.0;
    let tar_info: Array2<f64> = 1e-3 * array![[18.0, 0.0, z_targ, 1.5]]; //,[-18.0, 0.0, z_targ, 1.5],[9.0, 0.0, z_targ, 1.5],[-9.0, 0.0, z_targ, 1.5],[0.0, 0.0, z_targ, 1.5],[0.0, 12.0, z_targ, 4.0],[0.0, -12.0, z_targ, 4.0]];
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



// fn main2() {

// //    let x = array![-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0];
// //    let x = Array::range(0.0, 1300.0, 1.0);
//     let x = [1.0,2.0,3.0];
//     let y = [5.0,6.0,7.0];
//     let z = x - y;

//     println!("{:?}", z);
// //    println!("{:?}", y);
// }
