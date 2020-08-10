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
that you encounter bugs, please send a bug report to `parras@mpg-garching.mpg.de`.

## Requirements
- [NIFTy7](https://gitlab.mpcdf.mpg.de/ift/NIFTy) branch `NIFTy_7`
- [ducc0](https://gitlab.mpcdf.mpg.de/mtr/ducc) branch `ducc0`
- python-casacore
