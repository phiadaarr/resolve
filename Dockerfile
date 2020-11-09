FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq && apt-get install -qq \
    git python3-pip casacore-dev python3-pytest-cov python3-matplotlib
RUN pip3 install scipy git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_7
RUN pip3 install git+https://gitlab.mpcdf.mpg.de/mtr/ducc.git
