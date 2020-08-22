# resolve

Resolve aims to be a general radio aperature synthesis algorithm.
It is based on Bayesian principles and formulated in the language of information field theory.
Its features include single-frequency imaging with either only a diffuse or a diffuse+point-like sky model as prior, single-channel antenna-based calibration with a regularization in temporal domain and w-stacking.

Resolve is in beta stage: You are more than welcome to test it and help to make it applicable.
In the likely case that you encounter bugs, please send a bug report to `parras@mpg-garching.mpg.de`.

## Installation
- Install nifty7, e.g. via:
```
pip3 install --user git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_7
```
- For reading [measurement sets](https://casa.nrao.edu/Memos/229.html), install [python-casacore](https://github.com/casacore/python-casacore).
- For reading data in HDF5 format, install [h5py](http://www.h5py.org/), e.g. via:
```
pip3 install --user h5py
```

## Related publications

- The variable shadow of M87* ([arXiv](https://arxiv.org/abs/2002.05218)).
- Unified radio interferometric calibration and imaging with joint uncertainty quantification ([doi](https://doi.org/10.1051/0004-6361/201935555), [arXiv](https://arxiv.org/abs/1903.11169)).
- Radio imaging with information field theory ([doi](https://doi.org/10.23919/EUSIPCO.2018.8553533), [arXiv](https://arxiv.org/abs/1803.02174v1)).
