# Universal Backprojection Algorithm for Photoacoustic Computed Tomography
Replication of the algorithm demonstrated in:

[Xu, Minghua, and Lihong V. Wang. "Universal back-projection algorithm for photoacoustic computed tomography." *Physical Review E* 71.1 (2005): 016706.](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.71.016706)

This is implemented in both Python and Rust. The py file operates with the base conda distribution. I've used the Rust crates `ndarray` and `fftw` among others. You may be required to install fortran/openblas to compile (e.g. `sudo apt-get install gfortran libopenblas-dev` on debian). To observe speedups, use `cargo build --release`.

![alt text](./figure_1.png)
