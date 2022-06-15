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

import pickle

import nifty8 as ift
import numpy as np

from ..irg_space import IRGSpace
from ..util import assert_sky_domain


def polarization_overview(sky_field, name=None, offset=None):
    # Rick Perley says: Q = 0 and U = 1 corresponds to a p.a. of +45 degrees,
    # so the polarization line should extend from bottom right to upper left. 
    # (positive rotation is CCW).

    # Wikipedia says (https://en.wikipedia.org/wiki/Stokes_parameters): Q = 0
    # and U = 1 corresponds to bottom left to upper right.

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    assert_sky_domain(sky_field.domain)
    pdom, tdom, fdom, sdom = sky_field.domain

    val = sky_field.val

    if tdom.size != 1:
        raise NotImplementedError

    fig, axs = plt.subplots(ncols=pdom.size-2+2, nrows=fdom.size, figsize=[10, 10])
    axs = list(np.ravel(axs))

    # Figure out limits for color bars
    vmin, vmax = {}, {}
    for pol in pdom.labels:
        foo = sky_field.val[pdom.label2index(pol)]
        if pol == "I":
            vmin[pol], vmax[pol] = np.min(foo), np.max(foo)
        else:
            lim = 0.1*np.max(np.abs(foo))
            vmin[pol], vmax[pol] = -lim, lim
    lin = linear_polarization(sky_field)
    # /Figure out limits for color bars

    for ii, ff in enumerate(fdom.coordinates):
        colorbar = ff == fdom.coordinates[-1]
        title = ff == fdom.coordinates[0]
        for pol in pdom.labels:
            if pol in ["U", "Q"]:
                continue
            axx = axs.pop(0)
            val = sky_field.val[pdom.label2index(pol), 0, ii]
            if pol == "I" and val.min() < 0:
                raise RuntimeError("Stokes I cannot be negative")
            _plot_single_freq(
                    axx,
                    ift.makeField(sdom, val),
                    title=f"Stokes {pol}" if title else "",
                    colorbar=colorbar,
                    offset=offset,
                    norm=LogNorm(vmin=vmin[pol], vmax=vmax[pol]) if pol == "I" else None,
                    cmap="inferno" if pol == "I" else "seismic",
                )

        loop_fdom = IRGSpace([ff])
        loop_dom = pdom, tdom, loop_fdom, sdom
        loop_sky = ift.makeField(loop_dom, sky_field.val[:, :, ii:ii+1])
        ang = polarization_angle(loop_sky).val[0, 0]
        lin = linear_polarization(loop_sky).val[0, 0]

        axx = axs.pop(0)
        if title:
            axx.set_title("Linear polarization")
        _plot_single_freq(
                axx,
                ift.makeField(sdom, lin),
                title=f"Linear polarization" if title else "",
                colorbar=colorbar,
                offset=offset,
                norm=LogNorm(vmin["I"], vmax["I"]),
                cmap="inferno",
            )
        axx = axs.pop(0)
        if title:
            axx.set_title("Magnetic field orientation")
        foo = plt.cm.hsv(((np.angle(np.exp(1j*ang)) / np.pi) % 1))
        foo[..., -1] = (100*(lin-np.min(lin))/(np.max(lin)-np.min(lin))).clip(None, 1)
        foo[..., -1] /= np.max(foo[..., -1])
        foo = np.transpose(foo, (1, 0, 2))
        norm = mpl.colors.Normalize(vmin=-90, vmax=90)
        im = axx.imshow(foo, cmap="hsv", origin="lower", norm=norm, extent=_extent(sdom, offset))
        if colorbar:
            from matplotlib.ticker import StrMethodFormatter
            plt.colorbar(im, orientation="horizontal", ax=axx, format=StrMethodFormatter("{x:.0f}°")).set_ticks([-90, -45, 0, 45, 90])


    plt.tight_layout()
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
        plt.close()


def _plot_single_freq(axx, field, title, colorbar=True, offset=None, **kwargs):
    import matplotlib.pyplot as plt

    assert len(field.shape) == 2
    im = axx.imshow(field.val.T, extent=_extent(field.domain, offset), origin="lower", **kwargs)
    axx.set_title(title)
    if colorbar:
        plt.colorbar(im, orientation="horizontal", ax=axx)


def _extent(sdom, offset=None):
    sdom = ift.DomainTuple.make(sdom)
    assert len(sdom) == 1
    sdom = sdom[0]
    nx, ny = sdom.shape
    dx, dy = sdom.distances
    xlim, ylim = nx*dx/2, ny*dy/2
    if offset is None:
        return [-xlim, xlim, -ylim, ylim]
    else:
        ox, oy = offset
        return [-xlim+ox, xlim+ox, -ylim+oy, ylim+oy]


def polarization_quiver(ax, sky_field, nquivers=100, pfrac=0.001, vmin=None, vmax=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    assert_sky_domain(sky_field.domain)
    pdom, tdom, fdom, sdom = sky_field.domain
    assert all((pol in pdom.labels) for pol in ["I", "Q", "U"])
    assert tdom.size == fdom.size == 1
    ang = polarization_angle(sky_field).val[0, 0]
    lin = linear_polarization(sky_field).val[0, 0]

    im = ax.imshow(lin.T, cmap="inferno", norm=LogNorm(vmin=vmin, vmax=vmax), origin="lower",
                   extent=_extent(sdom))
    scale = np.max(lin)*max(lin.shape) * 5

    ang = np.ma.masked_where(lin < pfrac * np.max(lin), ang)
    lin = np.ma.masked_where(lin < pfrac * np.max(lin), lin)

    skip = max(sdom.shape) // nquivers
    ang = ang[::skip, ::skip]
    lin = lin[::skip, ::skip]

    nx, ny = sdom.shape
    xs = np.linspace(*ax.get_xlim(), ang.shape[0])
    ys = np.linspace(*ax.get_ylim(), ang.shape[1])
    dst = min(sdom.distances)
    ax.quiver(*np.meshgrid(xs, ys, indexing="ij"),
              np.cos(ang), np.sin(ang),
              angles="uv", pivot='mid',
              headaxislength=0, headwidth=0, headlength=0,
              scale=1.1/dst/skip, scale_units="xy",
              alpha=.6,
              )
    plt.colorbar(im, ax=ax, orientation="horizontal", label="Linear polarized flux [Jy/sr]")


def polarized_fraction(ax, sky_field, pfrac=0.001, vmax=None):
    import matplotlib.pyplot as plt

    assert_sky_domain(sky_field.domain)
    pdom, tdom, fdom, sdom = sky_field.domain
    assert all((pol in pdom.labels) for pol in ["I", "Q", "U"])
    assert tdom.size == fdom.size == 1
    lin = linear_polarization(sky_field).val[0, 0]
    total_int = sky_field.val[pdom.label2index("I"), 0, 0]

    frac = lin/total_int
    frac = np.ma.masked_where(lin < pfrac * np.max(lin), frac)

    im = ax.imshow(frac.T, cmap="viridis_r", origin="lower", vmin=0, vmax=vmax,
                   extent=_extent(sdom))

    plt.colorbar(im, ax=ax, orientation="horizontal", label="Polarized fraction [1]")


def polarization_angle(sky_field, faradaycorrection=0):
    assert_sky_domain(sky_field.domain)
    pdom, tdom, fdom, sdom = sky_field.domain
    u = sky_field.val[pdom.label2index("U")]
    q = sky_field.val[pdom.label2index("Q")]
    res = 0.5 * np.arctan2(u, q) - faradaycorrection
    # np.arctan2(u, q) = np.angle(q+1j*u)
    return ift.makeField((tdom, fdom, sdom), res)


def linear_polarization(sky_field):
    assert_sky_domain(sky_field.domain)
    pdom, tdom, fdom, sdom = sky_field.domain
    u = sky_field.val[pdom.label2index("U")]
    q = sky_field.val[pdom.label2index("Q")]
    res = np.sqrt(u ** 2 + q ** 2)
    return ift.makeField((tdom, fdom, sdom), res)
