# How to run the demos

## Basic imaging with automatic weighting

- Download the [data](https://www.philipp-arras.de/assets/CYG-ALL-2052-2MHZ.ms.tar.gz) and unpack it.
- Change the path under `[data]` in `cygnusa.cfg` to the path where the data is located.
- Run
```sh
python3 imaging_with_automatic_weighting.py cygnusa.cfg
```
or, if you have `mpi4py` installed:

``` sh
mpirun -np <ntasks> python3 imaging_with_automatic_weighting.py cygnusa.cfg
```
which should speed up the computation. The number of threads used *per task* can be set in the configuration file `cygnusa.cfg` in the section `[technical]/nthreads`. The number threads multiplied by the number of MPI tasks should not exceed the number CPU cores available on the machine.
