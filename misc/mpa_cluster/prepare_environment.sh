#!/bin/bash

# Run with ./prepare_environment <qname> <resolve_branch> <nifty_branch>
set -e

source $HOME/.bashrc

python3 -m venv venv$1
source venv$1/bin/activate
pip3 install matplotlib scipy python-casacore astropy h5py mpi4py pybind11
git clone -b $3 gitlab:ift/nifty
cd nifty && pip3 install -e . && cd ..
git clone -b $2 gitlab:parras/resolve
cd resolve && pip3 install -e . && cd ..
pip3 install --no-binary ducc0 ducc0
