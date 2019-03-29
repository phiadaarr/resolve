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
# Copyright(C) 2019 Max-Planck-Society

import numpy as np

import nifty5 as ift


def _wscreen(npix, domain, w):
    dst = domain.distances[0]
    dc = (np.linspace(start=-npix/2, stop=npix/2 - 1, num=npix)*dst)**2
    ls = np.broadcast_to(dc, (dc.shape[0],)*2)
    theta = np.sqrt(ls + ls.T)
    n = np.cos(theta)
    wscreen = np.exp(2*np.pi*1j*w*(n - 1))/n
    return ift.from_global_data(domain, wscreen)


def Response(dh, wplanes, domain):
    dst = domain.distances[0]
    npix = domain.shape[0]
    nrows = dh.vis.shape[0]
    nch = dh.vis.shape[1]

    # Prepare uv for RadioResponse. This class assumes that the harmonic
    # space has volume one. The NIFTy harmonic space has a non-trivial
    # volume which depends on the number of pixels and the field of view.
    # Therefore, uv needs to be rescaled.
    uv = dh.uv*dst
    w = dh.w

    uv = uv[~dh.flag]
    w = w[~dh.flag]

    mi, ma = w.min(), w.max()
    delta = (ma - mi)/wplanes
    wpl = np.linspace(mi, ma - delta, wplanes)  # left side of the bin
    s, ops = 0, []
    if wplanes > 1:
        for ii, ww in enumerate(wpl):
            wscreen = ift.makeOp(_wscreen(npix, domain, ww))
            last_plane = ii == wplanes - 1
            mask = np.logical_and(
                w >= ww, w <= ww + delta if last_plane else w < ww + delta)
            ss = np.sum(mask)
            if ss == 0:
                print('W-plane #{} is empty.'.format(ii))
                continue
            s += ss
            nfft = ift.NFFT(domain, uv[mask]).scale(domain.scalar_dvol)
            re = ift.Realizer(wscreen.target)
            mask = ift.from_global_data(ift.UnstructuredDomain(w.shape), ~mask)
            mask = ift.MaskOperator(mask).adjoint
            ops.append(mask @ nfft @ wscreen @ re)
        op = ift.utilities.my_sum(ops)
        print('{} w-planes are active.'.format(len(ops)))
        if s != len(w):
            raise RuntimeError('Bug detected.')
    else:
        nfft = ift.NFFT(domain, uv).scale(domain.scalar_dvol)
        re = ift.Realizer(nfft.domain)
        op = nfft @ re

    tgt = ift.DomainTuple.make((ift.UnstructuredDomain(nrows),
                                ift.UnstructuredDomain(nch)))
    mask = ift.from_global_data(tgt, dh.flag)
    mask = ift.MaskOperator(mask).adjoint
    pol_expander = .5*ift.ContractionOperator(dh.vis.domain, 2).adjoint
    return pol_expander @ mask @ op
