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
# Copyright(C) 2021-2022 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

from ..util import assert_sky_domain
from ..ubik_tools.plot_sky_hdf5 import _optimal_subplot_distribution
import numpy as np


def plot_dirty(fld, file_name):
    import matplotlib.pyplot as plt

    assert_sky_domain(fld.domain)
    pdom, tdom, fdom, sdom = fld.domain
    
    fig, axs = plt.subplots(**_optimal_subplot_distribution(pdom.size*tdom.size*fdom.size))
    axs = list(np.array(axs).ravel())
    for kk in pdom.labels:
        for itime, time in enumerate(tdom.coordinates):
            for ifreq, freq in enumerate(fdom.coordinates):
                axx = axs.pop(0)
                myarr = fld.val[pdom.label2index(kk), itime, ifreq]
                assert myarr.ndim == 2
                halfrange = np.max(np.abs(myarr))
                if halfrange == 0.:
                    continue
                norm = CenteredNorm(vcenter=0., halfrange=halfrange)
                im = axx.imshow(myarr.T, origin="lower", norm=norm, cmap="seismic")
                plt.colorbar(im, ax=axx)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_sky(fld, file_name):
    import matplotlib.pyplot as plt
    from matplotlib.colors import CenteredNorm, LogNorm

    assert_sky_domain(fld.domain)
    pdom, tdom, fdom, sdom = fld.domain
    
    fig, axs = plt.subplots(**_optimal_subplot_distribution(pdom.size*tdom.size*fdom.size))
    axs = list(np.array(axs).ravel())
    for kk in pdom.labels:
        polarr = fld.val[pdom.label2index(kk)]
        if kk == "I":
            norm = LogNorm(vmin=np.min(polarr), vmax=np.max(polarr))
        else:
            norm = CenteredNorm(vcenter=0., halfrange=np.max(np.abs(polarr)))
        for itime, time in enumerate(tdom.coordinates):
            for ifreq, freq in enumerate(fdom.coordinates):
                axx = axs.pop(0)
                im = axx.imshow(polarr[itime, ifreq].T, origin="lower", norm=norm)
                plt.colorbar(im, ax=axx)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
