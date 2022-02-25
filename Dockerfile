FROM debian:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq && apt-get install -qq git

# Actual dependencies
RUN apt-get update -qq && apt-get install -qq python3-pip casacore-dev pybind11-dev
RUN pip3 install scipy git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@likelihood_energies ducc0 matplotlib pybind11
# Optional dependencies
RUN pip3 install astropy jax jaxlib
RUN apt-get install -qq python3-mpi4py python3-h5py
# Testing dependencies
RUN apt-get install -qq python3-pytest-cov
# Documentation dependencies
RUN pip3 install sphinx pydata-sphinx-theme

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser
