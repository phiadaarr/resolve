FROM kernsuite/base:dev
ENV DEBIAN_FRONTEND noninteractive
RUN docker-apt-install dysco python3-casacore
RUN apt-get update -qq && apt-get install -qq python3-pip python3-matplotlib git
RUN pip3 install scipy pybind11 git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_7
RUN pip3 install git+https://gitlab.mpcdf.mpg.de/mtr/ducc.git@ducc0
RUN pip3 install git+https://gitlab.mpcdf.mpg.de/ift/resolve.git
RUN mkdir /work
