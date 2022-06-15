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
# Copyright(C) 2022 Max-Planck-Society
# Author: Philipp Arras

import argparse
import os
import pickle
from warnings import warn

import nifty8 as ift
import numpy as np
# For eval() spaces
from nifty8 import DomainTuple, MultiDomain
from nifty8.domains import *

from ..irg_space import IRGSpace
from ..polarization_space import PolarizationSpace
from ..sky_model import assert_sky_domain, default_sky_domain
from .fits import field2fits
# /For eval() spaces

def cmdline_visualize_sky_hdf5():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_name')
    parser.add_argument('what', help="Can be 'mean', 'stddev', 'stddev/mean', 'mean/stddev', "
                                      "'sample[i]'")
    parser.add_argument('stokes', help="Can be 'I', 'Q', 'U', 'V', 'polarizatedfraction' "
                                       "or nothing if output file ends with '.fits'. Then all "
                                       "Stokes parameters are written to the fits file.", nargs="?")
    parser.add_argument('--norm', help="Can be 'log', 'symmetric', 'linear'", default='linear')
    parser.add_argument('--vmin', type=float)
    parser.add_argument('--vmax', type=float)
    parser.add_argument('-o')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    return visualize_sky_hdf5(hdf5_file=args.file_name,
                              output_file=args.o,
                              what=args.what,
                              stokes=args.stokes,
                              norm=args.norm, vmin=args.vmin, vmax=args.vmax,
                              dpi=args.dpi)


def visualize_sky_hdf5(hdf5_file, output_file, what, stokes, norm="linear", vmin=None, vmax=None,
                       dpi=300):
    """

    Parameters
    ----------
    hdf5_file : str

    output_file : str

    what : str
        Can be 'mean', 'stddev', 'stddev/mean', 'mean/stddev', 'sample[i]'.

    stokes : str
        Can be 'I', 'Q', 'U', 'V', 'polarizatedfraction' or nothing if output
        file ends with '.fits'. Then all Stokes parameters are written to the
        fits file.

    norm : str
        Can be 'log', 'symmetric', 'linear', default is 'linear'.

    vmin : 

    vmax : 

    dpi : int
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("Need to install the optional dependency h5py")

    with h5py.File(hdf5_file, "r") as f:
        if what == "mean":
            if "stats" in f.keys() and "mean" in f["stats"].keys():
                arr = f["stats"]["mean"]
            elif len(f["samples"]) == 1:
                arr = f["samples"]["0"]
            else:
                raise RuntimeError
        elif what == "stddev":
            arr = f["stats"]["standard deviation"]
        elif what == "stddev/mean":
            arr = np.array(f["stats"]["standard deviation"])
            arr /= np.array(f["stats"]["mean"])
        elif what == "mean/stddev":
            arr = np.array(f["stats"]["mean"])
            arr /= np.array(f["stats"]["standard deviation"])
        elif what[:6] == "sample":
            arr = f["samples"][what[6:]]
        else:
            raise RuntimeError()

        fits = output_file is not None and os.path.splitext(output_file)[1] == ".fits"

        if "nifty domain" not in f.attrs and arr.shape[0] == 1:
            # Generate dummy domain if no proper domain information is present
            warn("Compatiblity mode")
            _, nt, nf, nx, ny = arr.shape
            dom = default_sky_domain(sdom=ift.RGSpace((nx, ny), (1, 1)),
                                     tdom=IRGSpace(np.arange(nt)),
                                     fdom=IRGSpace(np.arange(nf)),
                                     )
        else:
            dom = eval(f.attrs["nifty domain"])

        ind = lambda k: dom[0].label2index(k)
        if stokes == None:
            arr = np.array(arr)
        elif stokes == "polarizedfraction":
            sc = ift.StatCalculator()
            for ss in f["samples"].values():
                I = np.array(ss[ind("I")])
                Q = np.array(ss[ind("Q")])
                U = np.array(ss[ind("U")])
                V = np.array(ss[ind("V")])
                sc.add(np.sqrt(Q*Q + U*U + V*V) / I)
            if what == "mean":
                arr = sc.mean
            elif what == "stddev":
                arr = np.sqrt(sc.var)
            elif what == "stddev/mean":
                arr = np.sqrt(sc.var) / sc.mean
            elif what == "mean/stddev":
                arr = sc.mean / np.sqrt(sc.var)
            else:
                raise NotImplementedError
            arr = arr[None]
            dom = ift.DomainTuple.make((PolarizationSpace("I"), *dom[1:]))  # Dummy polarization space
        else:
            arr = np.array(arr[ind(stokes)])[None]
            dom = ift.DomainTuple.make((PolarizationSpace(stokes), *dom[1:]))

        assert isinstance(arr, np.ndarray)
        assert_sky_domain(dom)

    fld = ift.makeField(dom, arr)

    if fits:
        field2fits(fld, output_file)
    else:
        if fld.shape[0] != 1:
            raise RuntimeError("Need to specify the polarization to be plotted")
        _plot_with_mpl(fld, norm, vmin, vmax, output_file, dpi)


def _plot_with_mpl(fld, norm, vmin, vmax, output_file, dpi):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import CenteredNorm, LogNorm, Normalize
    except ImportError:
        from warnings import warn
        warn("Matplotlib could not be imported. Skip the plot")

    # Assume that only one polarization is present and get rid of the polarization dimension
    assert fld.shape[0] == 1
    fld = ift.ContractionOperator(fld.domain, 0)(fld)

    if norm == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
        cmap = "inferno"
    elif norm == "symmetric":
        norm = CenteredNorm(vcenter=0., halfrange=vmax)
        cmap = "seismic"
    elif norm == "linear":
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = "inferno"

    nt, nf, nx, ny = fld.shape

    fig, axs = plt.subplots(**_optimal_subplot_distribution(nt*nf))
    axs = list(np.array([axs]).ravel())

    tdom, fdom, sdom = fld.domain

    for ii, time in enumerate(tdom.coordinates):
        for jj, freq in enumerate(fdom.coordinates):
            axx = axs.pop(0)
            im = axx.imshow(fld.val[ii, jj].T, origin="lower", norm=norm, cmap=cmap)
            s = ""
            if len(tdom.coordinates) > 1:
                s += f"Time {time} "
            if len(fdom.coordinates) > 1:
                s += f"Frequency {freq} "
            if len(s) > 0:
                axx.set_title(s)
            plt.colorbar(im, ax=axx)
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=dpi)
    plt.close()


def _optimal_subplot_distribution(n):
    ny = int(np.ceil(np.sqrt(n)))
    nx = int(np.ceil(n/ny))
    assert nx*ny >= n
    return {"nrows": ny, "ncols": nx, "figsize": (6*nx, 6*ny)}
