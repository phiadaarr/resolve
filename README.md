# resolve

Resolve aims to be a general radio aperature synthesis algorithm. It is based on
Bayesian principles and formulated in the language of information field theory.
Its features include single-frequency imaging with either only a diffuse or a
diffuse+point-like sky model as prior, single-channel antenna-based calibration
with a regularization in temporal domain and w-stacking.

Resolve is in beta stage: You are more than welcome to test it and help to make
it applicable. Resolve is published in chunks whenever a new publication is
finished. If you would like to try resolve, please make sure to ask
`parras@mpa-garching.mpg.de` for the latest private version. In the likely case
that you encounter bugs, please send a bug report to parras@mpg-garching.mpg.de.

## Requirements
- [NIFTy5](https://gitlab.mpcdf.mpg.de/ift/NIFTy) tag `v5.0.1`
- [pyNFFT](https://pypi.python.org/pypi/pyNFFT)
- python-casacore

## Reproduce plots in paper
In order to reproduce the results of the latest resolve paper please contact
`parras@mpa-garching.mpg.de` for the data. All scripts needed for the plots in
the paper can be found in `scripts_paper/`.
