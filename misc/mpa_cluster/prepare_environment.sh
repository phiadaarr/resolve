#!/bin/bash

# Run with ./prepare_environment <base_folder> <qname> <resolve_branch> <nifty_branch>
set -e

module load python3

mkdir -p $1
cd $1

echo "Install venv to $1"

python3 -m venv venv$2
source venv$2/bin/activate
pip3 install matplotlib scipy python-casacore astropy h5py mpi4py pybind11
pip3 install git+https://gitlab.mpcdf.mpg.de/ift/nifty@$4

rm -rf resolve
git clone --recursive -b $3 gitlab:parras/resolve
cd resolve && pip3 install . && cd ..
rm -rf resolve

#pip3 install --no-binary ducc0 ducc0
pip3 install ducc0
