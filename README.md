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
- For reading in measurement sets, install [python-casacore](https://github.com/casacore/python-casacore).
- For reading data in HDF5 format, install [h5py](http://www.h5py.org/), e.g. via:
```
pip3 install --user h5py
```
