FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq && apt-get install -qq git

# Actual dependencies
RUN apt-get update -qq && apt-get install -qq python3-pip casacore-dev python3-matplotlib
RUN pip3 install scipy git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@empty_sample_lists ducc0
# Optional dependencies
RUN pip3 install astropy
RUN apt-get install -qq python3-mpi4py
# Testing dependencies
RUN apt-get install -qq python3-pytest-cov
# Documentation dependencies
RUN pip3 install sphinx pydata-sphinx-theme
RUN pip3 install jax jaxlib

# Create user (openmpi does not like to be run as root)
RUN useradd -ms /bin/bash testinguser
USER testinguser
WORKDIR /home/testinguser
