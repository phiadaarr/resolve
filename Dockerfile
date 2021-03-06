FROM debian:latest

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq && apt-get install -qq git

# Actual dependencies
RUN apt-get update -qq && apt-get install -qq python3-pip
RUN pip3 install git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_8 pybind11
# Optional dependencies
RUN pip3 install astropy jax jaxlib
RUN apt-get install -qq python3-mpi4py
# Testing dependencies
RUN apt-get install -qq python3-pytest-cov
# Documentation dependencies
RUN pip3 install sphinx pydata-sphinx-theme

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser
