# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import functools

try:
    from mpi4py import MPI
    master = MPI.COMM_WORLD.Get_rank() == 0
    comm = MPI.COMM_WORLD
    ntask = comm.Get_size()
    rank = comm.Get_rank()
    master = rank == 0
    mpi = ntask > 1
except ImportError:
    master = True
    mpi = False
    comm = None
    rank = 0


def onlymaster(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not master:
            return
        return func(*args, **kwargs)
    return wrapper
