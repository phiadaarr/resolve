# resolve

Documentation:
[http://ift.pages.mpcdf.de/resolve](http://ift.pages.mpcdf.de/resolve)

Resolve aims to be a general radio aperature synthesis algorithm.  It is based
on Bayesian principles and formulated in the language of information field
theory.  Its features include single-frequency imaging with either only a
diffuse or a diffuse+point-like sky model as prior, single-channel antenna-based
calibration with a regularization in temporal domain and w-stacking.

Resolve is in beta stage: You are more than welcome to test it and help to make
it applicable.  In the likely case that you encounter bugs, please contact me
via [email](mailto:c@philipp-arras.de).

## Requirements

For running the installation script:

- Python version 3.7 or later.
- C++17 capable compiler, e.g. g++ 7 or later.
- pybind11>=2.6
- setuptools

Automatically installed by installation script:

- ducc0
- nifty8
- numpy

Optional dependencies:

- astropy
- pytest, pytest-cov (for testing)
- mpi4py
- python-casacore (for reading measurement sets)
- h5py
- matplotlib

## Installation

For a blueprint how to install resolve, you may look at the [Dockerfile](./Dockerfile).

For installing resolve on a Linux machine, the following steps are necessary.
First install the necessary dependencies, for example via:

    pip3 install --upgrade pybind11 setuptools

Finally, clone the resolve repository and install resolve on your system:

    git clone https://gitlab.mpcdf.mpg.de/ift/resolve
    cd resolve
    python3 setup.py install --user

## Related publications

- The variable shadow of M87* ([arXiv](https://arxiv.org/abs/2002.05218)).
- Unified radio interferometric calibration and imaging with joint uncertainty quantification ([doi](https://doi.org/10.1051/0004-6361/201935555), [arXiv](https://arxiv.org/abs/1903.11169)).
- Radio imaging with information field theory ([doi](https://doi.org/10.23919/EUSIPCO.2018.8553533), [arXiv](https://arxiv.org/abs/1803.02174v1)).
