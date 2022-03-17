# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2021 Max-Planck-Society
# Author: Philipp Arras

import nifty8 as ift
import numpy as np


class AllreduceSum(ift.Operator):
    def __init__(self, oplist, comm):
        # FIXME Currently all operators in the oplist need to have the same
        # domain and target. In NIFTy, we generally support the addition of
        # operators that have different domains (not all keys need to be present
        # in all domains). This could be done here as well.

        # FIXME If all operators in oplist are linear operators, we could
        # automatically instantiate AllreduceSumLinear.

        self._oplist, self._comm = oplist, comm
        self._domain = ift.makeDomain(
            _get_global_unique(oplist, lambda op: op.domain, comm)
        )
        self._target = ift.makeDomain(
            _get_global_unique(oplist, lambda op: op.target, comm)
        )

    def apply(self, x):
        self._check_input(x)
        opx = [op(x) for op in self._oplist]
        if not ift.is_linearization(x):
            return ift.utilities.allreduce_sum(opx, self._comm)
        val = ift.utilities.allreduce_sum([lin.val for lin in opx], self._comm)
        jac = AllreduceSumLinear([lin.jac for lin in opx], self._comm)
        if _get_global_unique(opx, lambda op: op.metric is None, self._comm):
            return x.new(val, jac)
        met = AllreduceSumLinear([lin.metric for lin in opx], self._comm)
        return x.new(val, jac, met)


class SliceSum(ift.Operator):
    """Sum Operator that slices along the first axis of the input field and
    computes the sum in parallel using MPI.
    """
    def __init__(self, oplist, index_low, parallel_space, comm):
        # FIXME if oplist contains only linear operators instantiate
        # SliceSumLinear instead
        self._oplist, self._comm = oplist, comm
        self._lo = int(index_low)
        assert len(parallel_space.shape) == 1
        if len(oplist) > 0:
            assert index_low < parallel_space.shape[0]
        else:
            assert index_low == parallel_space.shape[0]
        doms = _get_global_unique(oplist, lambda op: op.domain, comm)

        if isinstance(doms, ift.MultiDomain):
            dom = {}
            for kk in doms.keys():
                dom[kk] = (parallel_space,) + tuple(dd for dd in doms[kk])
        else:
            dom = (parallel_space,) + tuple(dd for dd in doms)
        self._domain = ift.makeDomain(dom)
        tgt = _get_global_unique(oplist, lambda op: op.target, comm)
        self._target = ift.makeDomain(tgt)
        self._paraspace = parallel_space

    def apply(self, x):
        self._check_input(x)
        if not ift.is_linearization(x):
            opx = []
            for ii, op in enumerate(self._oplist):
                if isinstance(x, ift.MultiField):
                    foo = {kk: vv.val[self._lo + ii] for kk, vv in x.items()}
                else:
                    foo = x.val[self._lo + ii]
                opx.append(op(ift.makeField(op.domain, foo)))
            return ift.utilities.allreduce_sum(opx, self._comm)
        oplin = []
        for ii, op in enumerate(self._oplist):
            if isinstance(x.domain, ift.MultiDomain):
                foo = {kk: vv.val[self._lo + ii] for kk, vv in x.val.items()}
            else:
                foo = x.val.val[self._lo + ii]
            foo = ift.makeField(op.domain, foo)
            oplin.append(op(ift.Linearization.make_var(foo, x.want_metric)))
        val = ift.utilities.allreduce_sum([lin.val for lin in oplin], self._comm)
        args = self._lo, self._paraspace, self._comm
        jac = SliceSumLinear([lin.jac for lin in oplin], *args)
        if _get_global_unique(oplin, lambda op: op.metric is None, self._comm):
            return x.new(val, jac)
        met = SliceLinear([lin.metric for lin in oplin], *args)
        return x.new(val, jac, met)


class SliceSumLinear(ift.LinearOperator):
    """Special case of AllreduceSumLinear @ Slicer"""
    def __init__(self, oplist, index_low, parallel_space, comm):
        assert all(isinstance(oo, ift.LinearOperator) for oo in oplist)
        doms = _get_global_unique(oplist, lambda op: op.domain, comm)
        if isinstance(doms, ift.MultiDomain):
            dom = {}
            for kk in doms.keys():
                dom[kk] = (parallel_space,) + tuple(dd for dd in doms[kk])
        else:
            dom = (parallel_space,) + tuple(dd for dd in doms)
        self._domain = ift.makeDomain(dom)
        tgt = _get_global_unique(oplist, lambda op: op.target, comm)
        self._target = ift.makeDomain(tgt)
        cap = _get_global_unique(oplist, lambda op: op._capability, comm)
        self._capability = (self.TIMES | self.ADJOINT_TIMES) & cap
        self._oplist = oplist
        self._comm = comm
        self._lo = index_low
        local_nwork = [len(oplist)] if comm is None else comm.allgather(len(oplist))
        self._nwork = sum(local_nwork)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if mode == self.TIMES:
            opx = []
            for ii, op in enumerate(self._oplist):
                if isinstance(x, ift.MultiField):
                    foo = {kk: vv.val[self._lo + ii] for kk, vv in x.items()}
                else:
                    foo = x.val[self._lo + ii]
                opx.append(op(ift.makeField(op.domain, foo)))
            return ift.utilities.allreduce_sum(opx, self._comm)
        else:
            lst = [op.adjoint(x).val for op in self._oplist]
            res = allgather_dispatch(lst, self._comm, self._nwork)
            return ift.makeField(self.domain, res)


class SliceLinear(ift.EndomorphicOperator):
    def __init__(self, oplist, index_low, parallel_space, comm):
        assert all(isinstance(oo, ift.LinearOperator) for oo in oplist)
        doms = _get_global_unique(oplist, lambda op: op.domain, comm)
        if isinstance(doms, ift.MultiDomain):
            dom = {}
            for kk in doms.keys():
                dom[kk] = (parallel_space,) + tuple(dd for dd in doms[kk])
        else:
            dom = (parallel_space,) + tuple(dd for dd in doms)
        self._domain = ift.makeDomain(dom)
        cap = _get_global_unique(oplist, lambda op: op._capability, comm)
        self._capability = (self.TIMES | self.ADJOINT_TIMES) & cap
        self._oplist = oplist
        self._comm = comm
        self._lo = index_low
        local_nwork = [len(oplist)] if comm is None else comm.allgather(len(oplist))
        self._nwork = sum(local_nwork)

    def apply(self, x, mode):
        self._check_input(x, mode)
        if isinstance(self.domain, ift.MultiDomain):
            out_list = []
            for ii, op in enumerate(self._oplist):
                inp = ift.makeField(op.domain, {kk: x.val[kk][self._lo + ii] for kk in self.domain.keys()})
                out_list.append(op.apply(inp, mode).val)
            res = dict_allgather(out_list, self._comm, self._nwork)
            return ift.makeField(self.domain, res)
        res = array_allgather(
            [
                op.apply(ift.makeField(op.domain, x.val[self._lo + ii]), mode).val
                for ii, op in enumerate(self._oplist)
            ],
            self._comm,
            self._nwork
        )
        return ift.makeField(self._domain, res)

    def draw_sample(self, from_inverse=False):
        sseq = ift.random.spawn_sseq(self._nwork)
        local_samples = []
        for ii, op in enumerate(self._oplist):
            with ift.random.Context(sseq[self._lo + ii]):
                local_samples.append(op.draw_sample(from_inverse).val)
        res = allgather_dispatch(local_samples, self._comm, self._nwork)
        return ift.makeField(self._domain, res)


class AllreduceSumLinear(ift.LinearOperator):
    def __init__(self, oplist, comm=None):
        assert all(isinstance(oo, ift.LinearOperator) for oo in oplist)
        self._domain = ift.makeDomain(
            _get_global_unique(oplist, lambda op: op.domain, comm)
        )
        self._target = ift.makeDomain(
            _get_global_unique(oplist, lambda op: op.target, comm)
        )
        cap = _get_global_unique(oplist, lambda op: op._capability, comm)
        self._capability = (self.TIMES | self.ADJOINT_TIMES) & cap
        self._oplist = oplist
        self._comm = comm
        local_nwork = [len(oplist)] if comm is None else comm.allgather(len(oplist))
        size, rank, _ = ift.utilities.get_MPI_params_from_comm(comm)
        self._nwork = sum(local_nwork)
        self._lo = ([0] + list(np.cumsum(local_nwork)))[rank]

    def apply(self, x, mode):
        self._check_input(x, mode)
        lst = [op.apply(x, mode) for op in self._oplist]
        return ift.utilities.allreduce_sum(lst, self._comm)

    def draw_sample(self, from_inverse=False):
        sseq = ift.random.spawn_sseq(self._nwork)
        local_samples = []
        for ii, op in enumerate(self._oplist):
            with ift.random.Context(sseq[self._lo + ii]):
                local_samples.append(op.draw_sample(from_inverse))
        return ift.utilities.allreduce_sum(local_samples, self._comm)


def _get_global_unique(lst, f, comm):
    caps = [f(oo) for oo in lst]
    if comm is not None:
        caps = comm.allgather(caps)
        caps = [aa for cc in caps for aa in cc]
    cap = caps[0]
    assert all(cc == cap for cc in caps)
    return cap


def allgather_dispatch(obj, comm, nwork):
    if isinstance(obj[0], np.ndarray):
        return array_allgather(obj, comm, nwork)
    if isinstance(obj[0], dict):
        return dict_allgather(obj, comm, nwork)


def array_allgather(arrs, comm, nwork):
    if comm is None:
        full_lst = np.array(arrs)
    else:
        from mpi4py import MPI
        size = comm.Get_size()
        send_buf = np.array(arrs)
        recv_buffer_shape = (nwork,) + send_buf.shape[1:]

        full_lst = np.empty(recv_buffer_shape)

        send_count = []
        displacement = [0]

        for rank in range(size):
            lo, hi = ift.utilities.shareRange(nwork, size, rank)
            n_work_per_rank = hi - lo
            send_count_per_rank = np.prod((n_work_per_rank,) + send_buf.shape[1:])
            send_count.append(send_count_per_rank)

            if rank != size - 1:
                displacement.append(send_count_per_rank + displacement[rank])

        comm.Allgatherv([send_buf, MPI.DOUBLE], [full_lst, tuple(send_count), tuple(displacement), MPI.DOUBLE])
    return full_lst


def dict_allgather(lst, comm, nwork):
    """Apply array_allgather for each key of a dictionary.

    Parameters
    ----------
    lst: list
        List of dictionaries. All dictionaries need to have the same keys.
    """

    keys = lst[0].keys()
    for aa in lst:
        assert isinstance(aa, dict)
        assert aa.keys() == keys
    return {kk: array_allgather([ll[kk] for ll in lst], comm, nwork) for kk in keys}
