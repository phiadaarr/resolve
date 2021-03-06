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
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

import os
import sys
from functools import partial, reduce
from operator import add
from time import time

import matplotlib
import matplotlib.pyplot as plt
import nifty8 as ift
import numpy as np
import resolve as rve

n_cpus = os.cpu_count()
check_equality = False

eberas = [
    "#1f77b4",
    "#ff7f0e",
    "#6ce06c",
    "#d62728",
    "#b487dd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]
matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=eberas)


def get_cpp_op(args_cfm, args_lst, args_zm, nthreads):
    cfm = rve.CorrelatedFieldMaker(**args_cfm, nthreads=nthreads)
    for aa in args_lst:
        cfm.add_fluctuations(**aa)
    cfm.set_amplitude_total_offset(**args_zm)
    return cfm.finalize(0)


def get_nifty_op(args_cfm, args_lst, args_zm, nthreads):
    cfm = ift.CorrelatedFieldMaker(**args_cfm)
    for aa in args_lst:
        cfm.add_fluctuations(**aa)
    cfm.set_amplitude_total_offset(**args_zm)
    op = cfm.finalize(0)
    ift.set_nthreads(nthreads)
    return op


def get_pspecs_op(args_cfm, args_lst, args_zm, nthreads):
    cfm = ift.CorrelatedFieldMaker(**args_cfm)
    for aa in args_lst:
        cfm.add_fluctuations(**aa)
    cfm.set_amplitude_total_offset(**args_zm)
    ampls = cfm.get_normalized_amplitudes()
    ift.set_nthreads(nthreads)
    return reduce(add, [aa.ducktape_left(str(ii)) for ii, aa in enumerate(ampls)])


def perf_nifty_operators(op_dct, name, domain_dtype=np.float64, ntries=1):
    xs = list(range(1, n_cpus + 1))

    def init(keys, n):
        return {kk: n * [None] for kk in keys}

    linestyles = list(matplotlib.lines.lineStyles.keys())
    if len(linestyles) < len(op_dct):
        raise RuntimeError(
            f"Too many operators, got {len(op_dct)}. Only {len(linestyles)} are supported."
        )

    args = op_dct.keys(), n_cpus
    times = init(*args)
    times_with_jac = init(*args)
    jac_times = init(*args)
    jac_adj_times = init(*args)

    for ii, nthreads in enumerate(xs):
        print(f"{nthreads} / {n_cpus}")
        for kk, oo in op_dct.items():
            oo = oo(nthreads)
            pos = ift.from_random(oo.domain, dtype=domain_dtype)
            lin = ift.Linearization.make_var(pos)

            t0 = time()
            for _ in range(ntries):
                res = oo(pos)
            times[kk][ii] = (time() - t0) / ntries

            t0 = time()
            for _ in range(ntries):
                reslin = oo(lin)
            times_with_jac[kk][ii] = (time() - t0) / ntries

            t0 = time()
            for _ in range(ntries):
                reslin.jac(pos)
            jac_times[kk][ii] = (time() - t0) / ntries

            t0 = time()
            for _ in range(ntries):
                reslin.jac.adjoint(res)
            jac_adj_times[kk][ii] = (time() - t0) / ntries

    for ys, mode in [
        (times, "op(fld)"),
        (times_with_jac, "op(lin)"),
        (jac_times, "Jacobian"),
        (jac_adj_times, "Adjoint Jacobian"),
    ]:
        c = None
        for iop, kk in enumerate(op_dct.keys()):
            if mode == "op(fld)":
                lbl = f"{mode} | {kk}"
            elif iop == 0:
                lbl = f"{mode}"
            else:
                lbl = None
            line = plt.plot(xs, ys[kk], label=lbl, color=c, linestyle=linestyles[iop])
            c = line[0].get_color()
    plt.xlabel("# threads")
    plt.ylabel("Wall time [s]")
    plt.xlim([0, None])
    plt.ylim([0, None])
    plt.title(name)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "quick":
        total_Ns = [2]
        dom0 = [ift.RGSpace(4)]
        dom1 = [ift.RGSpace(5)]
        name = ["quick"]
        ntries_lst = [1]
    else:
        total_Ns = [1, 58, 200, 4, 1]
        dom0 = [
            ift.RGSpace([2000, 2000]),
            ift.RGSpace(1120),
            ift.RGSpace([42, 42], [0.1, 0.1]),
            ift.RGSpace([4000, 4000], [2e-6, 2e-6]),
            ift.RGSpace([500, 500]),
        ]
        dom1 = [None, ift.RGSpace(200), ift.RGSpace(88, 0.893), None, ift.RGSpace(30)]
        name = [
            "single freq imaging",
            "meerkat_calibration",
            "jroth_calibration",
            "polarization_imaging",
            "eht",
        ]
        ntries_lst = [10, 1, 1, 1, 2]

    for d0, d1, total_N, nm, ntries in zip(dom0, dom1, total_Ns, name, ntries_lst):
        print(f"Working on {nm}")
        dofdex = list(range(total_N))
        args_cfm = dict(prefix="", total_N=total_N)
        args_lst = []
        if d0 is not None:
            args_lst.append(
                dict(
                    target_subdomain=d0,
                    fluctuations=(1.0, 1.0),
                    flexibility=(2.0, 2),
                    asperity=(0.1, 0.1),
                    loglogavgslope=(-2, 0.1),
                    prefix="dom0",
                    dofdex=dofdex,
                )
            )
        if d1 is not None:
            args_lst.append(
                dict(
                    target_subdomain=d1,
                    fluctuations=(2.0, 0.1),
                    flexibility=(1.0, 2),
                    asperity=(0.2, 0.1),
                    loglogavgslope=(-3, 0.321),
                    prefix="dom1",
                    dofdex=dofdex,
                )
            )
        args_zm = dict(offset_mean=1.2, offset_std=(1.0, 0.2), dofdex=dofdex)

        operators = {
            "NIFTy": partial(get_nifty_op, args_cfm, args_lst, args_zm),
            "resolve_support": partial(get_cpp_op, args_cfm, args_lst, args_zm),
            "pspecs": partial(get_pspecs_op, args_cfm, args_lst, args_zm),
        }
        if check_equality:
            rve.operator_equality(operators["NIFTy"](1), operators["resolve_support"](1))
        perf_nifty_operators(operators, nm, ntries=ntries)
