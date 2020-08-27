FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq && apt-get install -qq \
    git python3-pip casacore-dev python3-pytest-cov python3-h5py python3-matplotlib
RUN pip3 install ducc0 scipy git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_7