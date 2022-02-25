# resolve

Documentation:
[http://ift.pages.mpcdf.de/resolve](http://ift.pages.mpcdf.de/resolve)

Resolve aims to be a general radio aperture synthesis algorithm.
It is based on Bayesian principles and formulated in the language of information field theory.
Its features include single-frequency imaging with either only a diffuse or a diffuse+point-like sky model as prior, single-channel antenna-based calibration with a regularization in temporal domain and w-stacking.

Resolve is in beta stage: You are more than welcome to test it and help to make it applicable.
In the likely case that you encounter bugs, please contact me via [email](mailto:parras@mpg-garching.mpg.de).

## Requirements

- Python >= 3.7
- [nifty8](https://gitlab.mpcdf.mpg.de/ift/nifty)
- [ducc0](https://gitlab.mpcdf.mpg.de/mtr/ducc)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [pybind11](https://github.com/pybind/pybind11)
- A C++17-capable compiler
- Optional dependencies are:
    - For reading [measurement sets](https://casa.nrao.edu/Memos/229.html), install [python-casacore](https://github.com/casacore/python-casacore).
    - For reading and writing FITS files: astropy.
    - Some operators support [jax](https://github.com/google/jax).

## Related publications

- The variable shadow of M87* ([arXiv](https://arxiv.org/abs/2002.05218)).
- Unified radio interferometric calibration and imaging with joint uncertainty quantification ([doi](https://doi.org/10.1051/0004-6361/201935555), [arXiv](https://arxiv.org/abs/1903.11169)).
- Radio imaging with information field theory ([doi](https://doi.org/10.23919/EUSIPCO.2018.8553533), [arXiv](https://arxiv.org/abs/1803.02174v1)).

## How to run the demos
### Basic imaging with automatic weighting

- Download the [data](https://www.philipp-arras.de/assets/CYG-ALL-2052-2MHZ.ms.tar.gz) and unpack it.
- Change the path under `[data]` in `cygnusa.cfg` to the path where the data is located.
- Run
```sh
resolve cygnusa.cfg
```
or, if you have `mpi4py` installed:

``` sh
mpirun -np <ntasks> resolve cygnusa.cfg
```
which should speed up the computation. The number of threads used *per task* can
be set via `-j`.  The number threads multiplied by the number of MPI tasks
should not exceed the number CPU cores available on the machine.

### Multi-frequency imaging
- Download the [data](https://www.philipp-arras.de/assets/mf_test_data.npz) and unpack it.
