FROM kernsuite/base:latest
ENV DEBIAN_FRONTEND noninteractive
RUN docker-apt-install dysco python3-casacore
RUN apt-get update -qq && apt-get install -qq python3-pip python3-matplotlib git
RUN pip3 install scipy pybind11 git+https://gitlab.mpcdf.mpg.de/ift/nifty.git@NIFTy_7
RUN pip3 install git+https://gitlab.mpcdf.mpg.de/mtr/ducc.git@ducc0
COPY . /resolve
RUN cd resolve && python3 setup.py install
ENTRYPOINT ["python3", "resolve/misc/ms2npz.py"]
