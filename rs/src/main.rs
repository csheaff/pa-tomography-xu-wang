use ndarray::{prelude::*, stack, Zip};
use ndarray_linalg::{norm::Norm, types::Scalar};
use ndarray_stats::QuantileExt; // this adds basic stat methods to your arrays
                                //use ndarray_stats::SummaryStatisticsExt;
use fftw::array::AlignedVec;
use fftw::plan::*;
use fftw::types::*;
use num_integer::Roots;
use std::f64::consts::PI;
use std::time::Instant;

fn fft_priv(x: &Array1<c64>, n: usize, sign: Sign) -> Array1<c64> {
    let mut xfft = AlignedVec::new(n);
    let mut xs_aligned = AlignedVec::new(n);
    for (x_aligned, &x) in xs_aligned.iter_mut().zip(x) {
        *x_aligned = x;
    }

    let mut plan: C2CPlan64 = C2CPlan::aligned(&[n], sign, Flag::MEASURE).unwrap();

    plan.c2c(&mut xs_aligned, &mut xfft).unwrap();
    Array1::from(Vec::from(xfft.as_slice()))
}

fn fft(x: &Array1<c64>, n: usize) -> Array1<c64> {
    // this is unnormalized, just like scipy.fftpack.fft

    fft_priv(x, n, Sign::Forward)
}

fn ifft(x: &Array1<c64>) -> Array1<c64> {
    // this will normalize, just like scipy.fftpack.ifft

    fft_priv(x, x.len(), Sign::Backward) / c64::new(x.len() as f64, 0.0)
}

fn step_fn(x: Array1<f64>) -> Array1<f64> {
    //0.5 * (x.mapv(f64::signum) + 1.0)
    x.mapv_into(|v| 0.5 * (v.signum() + 1.0))
}

fn meshgrid_3d(
    x1: &Array1<f64>,
    x2: &Array1<f64>,
    x3: &Array1<f64>,
) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
    let mut xx = Array3::<f64>::zeros((x2.len(), x1.len(), x3.len()));
    let mut yy = xx.clone();
    let mut zz = xx.clone();
    for m in 0..x2.len() {
        for n in 0..x3.len() {
            let mut slice = xx.slice_mut(s![m, .., n]);
            slice.assign(&x1);
        }
    }
    for m in 0..x1.len() {
        for n in 0..x3.len() {
            let mut slice = yy.slice_mut(s![.., m, n]);
            slice.assign(&x2);
        }
    }
    for m in 0..x2.len() {
        for n in 0..x1.len() {
            let mut slice = zz.slice_mut(s![m, n, ..]);
            slice.assign(&x3);
        }
    }
    (xx, yy, zz)
}

fn array_indexing_3d_complex(x: &Array1<c64>, ind: &Array3<usize>) -> Array3<c64> {
    Zip::from(ind).apply_collect(|idx| x[*idx])
}

fn get_signals(tar_info: &ArrayView1<f64>, xd: &Array1<f64>, t: &Array1<f64>) -> Array3<f64> {
    let det_len = 2e-3;
    let n_subdet = 25;
    let n_subdet_perdim = n_subdet.sqrt();
    let subdet_pitch = det_len / (n_subdet as f64).sqrt();
    let subdet_ind =
        Array::range(0.0, n_subdet_perdim as f64, 1.0) - (n_subdet_perdim as f64 - 1.0) / 2.0;
    let subdet_offset = subdet_pitch * &subdet_ind;
    let fs = 1.0 / (t[1] - t[0]);
    let fc = 4e6;
    let c = 1484.0;
    let n_det_x = xd.len();
    let tar_rad = tar_info[3];
    let tar_xyz = tar_info.slice(s![..3]);
    let ct = c * t;

    let mut sigs = Array3::<f64>::zeros((n_det_x, n_det_x, t.len()));
    for (xi, &x) in xd.iter().enumerate() {
        for (yi, &y) in xd.iter().enumerate() {
            let mut pa_sig = Array1::<f64>::zeros(t.len());
            for offset_x in subdet_offset.iter() {
                for offset_y in subdet_offset.iter() {
                    let det_xyz = array![x + offset_x, y + offset_y, 0.0];
                    let r = (det_xyz - tar_xyz).norm_l2();
                    let step_fn_arg = ct.mapv(|v| tar_rad - (r - v).abs());
                    pa_sig = pa_sig + step_fn(step_fn_arg) * (r - &ct) / (2.0 * r);
                }
            }
            let pr = pa_sig / n_subdet as f64;
            let mut slice = sigs.slice_mut(s![xi, yi, ..]);
            slice.assign(&pr);
        }
    }
    sigs
}

fn perf_tom(
    sigs: &Array3<f64>,
    xd: &Array1<f64>,
    t: &Array1<f64>,
    z_targ: f64,
) -> (Array3<f64>, Array1<f64>, Array1<f64>, Array1<f64>) {
    let c = 1484.0;
    let res = 500e-6;
    let xf = Array::range(xd[0], xd[xd.len() - 1] + res, res);
    let yf = xf.clone();
    let zf = array![z_targ];
    let (Yf, Xf, Zf) = meshgrid_3d(&yf, &xf, &zf); // DOUBLE-CHECK
    let Z2 = Zf.mapv(|Zf| Zf.powi(2));
    let fs = 1.0 / (t[1] - t[0]);
    let nfft = 2048;
    let fv = (fs / 2.0) * Array::linspace(0.0, 1.0, nfft / 2 + 1);
    let fv2 = -1.0 * (fs / 2.0) * Array::linspace(1.0, 0.0, nfft / 2 + 1);
    let fv2_slice = fv2.slice(s![1..(fv2.len() - 1)]);
    let fv3 = stack![Axis(0), fv, fv2_slice];
    let k = 2.0 * PI * fv3 / c;
    let ds = (2e-3).powi(2);
    let pf = 0.0 * &k; // make zeros array with length k.len()
    let mut pnum = Array3::<c64>::zeros(Yf.raw_dim());
    let mut pden = Array3::<c64>::zeros(Yf.raw_dim());
    let yd = xd.clone();
    let k = k.mapv(|x| c64::new(x, 0.0)); // convert to complex
    let t = t.mapv(|x| c64::new(x, 0.0)); // convert to complex
    for xi in 0..xd.len() {
        let X2 = &Xf - xd[xi];
        let X2 = X2.mapv(|X2| X2.powi(2));
        for yi in 0..yd.len() {
            let Y2 = &Yf - yd[yi];
            let Y2 = Y2.mapv(|Y2| Y2.powi(2));
            let dist2 = &X2 + &Y2 + &Z2;
            let dist = &dist2.mapv(f64::sqrt);
            let distind = (fs / c) * dist;
            let distind = distind.mapv(|x| <f64>::round(x) as usize);
            let p = sigs.slice(s![xi, yi, ..]).to_owned();
            let p = p.mapv(|x| c64::new(x, 0.0)); // convert to complex
            let p_w = fft(&p, nfft);
            let p_filt_w = c64::new(0.0, -1.0) * &k * p_w;
            let p_filt = ifft(&p_filt_w);
            let b =
                c64::new(2.0, 0.0) * &p - c64::new(2.0, 0.0) * &t * c * p_filt.slice(s![..p.len()]);
            let b1 = array_indexing_3d_complex(&b, &distind);
            let omega = (ds / dist2) * &Zf / dist;
            let omega = omega.mapv(|x| c64::new(x, 0.0)); // convert to complex
            pnum = pnum + &omega * &b1;
            pden += &omega;
        }
        //    	println!("Reconstrucing image with detector row {}", xd.len() - xi);
    }
    let pg = pnum / pden;
    let pg_max_ind = pg.mapv(|x| x.norm()).argmax().unwrap(); // index of maximum magnitude
    let pg_max = pg[pg_max_ind];
    let pfnorm = (pg / pg_max).mapv(|x| x.re());
    (pfnorm, xf, yf, zf)
}

fn tom_plot(pfnorm: &Array3<f64>, xf: &Array1<f64>, yf: &Array1<f64>, dr: f64) {
    let pfnormlog = 20.0 * pfnorm.mapv(|x| x.abs().log10());
    let pfnormlog = pfnormlog.mapv(|x| x.max(-dr));
    let pfnormlog = 255.0 * (pfnormlog + dr) / dr;
    let pfnormlog = pfnormlog.mapv(|x| x as u8);
    let imgx = pfnormlog.shape()[0] as u32;
    let imgy = pfnormlog.shape()[1] as u32;
    let pfnormlog = pfnormlog.slice(s![.., .., 0]).to_owned().into_raw_vec(); // get plane
    let imgbuf = image::GrayImage::from_vec(imgx, imgy, pfnormlog);
    imgbuf.unwrap().save("pfnormlog-rs.png");
}

fn main() {
    let before = Instant::now();
    let z_targ = 15.0;
    let tar_info: Array2<f64> = 1e-3
        * array![
            [18.0, 0.0, z_targ, 1.5],
            [-18.0, 0.0, z_targ, 1.5],
            [9.0, 0.0, z_targ, 1.5],
            [-9.0, 0.0, z_targ, 1.5],
            [0.0, 0.0, z_targ, 1.5],
            [0.0, 12.0, z_targ, 4.0],
            [0.0, -12.0, z_targ, 4.0]
        ];
    let z_targ = z_targ * 1e-3;
    let n_targ = tar_info.shape()[0];
    let aper_len = 60e-3;
    let det_pitch = (2.0 / 3.0) * 1e-3;
    let xd = Array::range(-aper_len / 2.0, aper_len / 2.0 + det_pitch, det_pitch);
    let n_det = xd.len();
    let fs = 20e6;
    let ts = 1.0 / fs;
    let t = Array::range(0.0, 65e-6 + ts, ts);
    let mut sigs = Array3::<f64>::zeros((n_det, n_det, t.len()));
    for n in 0..n_targ {
        println!(
            "Generating recorded signals arising from target {} of {}",
            n + 1,
            n_targ
        );
        let ti_slice = tar_info.slice(s![n, ..]);
        sigs = sigs + get_signals(&ti_slice, &xd, &t);
    }
    println!("Reconstructing targets");
    let (pfnorm, xf, yf, zf) = perf_tom(&sigs, &xd, &t, z_targ);
    println!(
        "Mean value of all recorded signals = {:?}",
        sigs.mean().unwrap()
    );
    println!(
        "Mean value of reconstructed volume = {:?}",
        pfnorm.mean().unwrap()
    );
    tom_plot(&pfnorm, &xf, &yf, 6.0);
    // Be sure to bench by running:
    // $ cargo run --release
    println!("Elapsed time: {:.2?} s", before.elapsed());
}
