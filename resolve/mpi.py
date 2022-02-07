# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Author: Philipp Arras

import functools

import nifty8 as ift

from .util import my_asserteq

try:
    from mpi4py import MPI

    master = MPI.COMM_WORLD.Get_rank() == 0
    comm = MPI.COMM_WORLD
    comm_self = MPI.COMM_SELF
    ntask = comm.Get_size()
    rank = comm.Get_rank()
    master = rank == 0
    mpi = ntask > 1

    if ntask == 1:
        master = True
        mpi = False
        comm = None
        comm_self = None
        rank = 0
except ImportError:
    master = True
    mpi = False
    comm = None
    comm_self = None
    rank = 0


def onlymaster(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not master:
            return
        state0 = ift.random.getState()
        f = func(*args, **kwargs)
        state1 = ift.random.getState()
        my_asserteq(state0, state1)
        return f

    return wrapper


def barrier(comm=None):
    if comm is None:
        return
    comm.Barrier()
