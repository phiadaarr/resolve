import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import LogNorm
import matplotlib as mpl
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import nifty8 as ift
import numpy as np
from ...irg_space import IRGSpace

from ...util import assert_sky_domain


def polarization_overview(sky_field, name=None, offset=None):
    # Rick Perley says: Q = 0 and U = 1 corresponds to a p.a. of +45 degrees,
    # so the polarization line should extend from bottom right to upper left. 
    # (positive rotation is CCW).

    # Wikipedia says (https://en.wikipedia.org/wiki/Stokes_parameters): Q = 0
    # and U = 1 corresponds to bottom left to upper right.

    assert_sky_domain(sky_field.domain)
    pdom, tdom, fdom, sdom = sky_field.domain

    val = sky_field.val

    if tdom.size != 1:
        raise NotImplementedError

    fig, axs = plt.subplots(nrows=pdom.size+1, ncols=fdom.size, figsize=[10, 10])
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
    # /Figure out limits for color bars

    for ii, ff in enumerate(fdom.coordinates):
        colorbar = ff == fdom.coordinates[-1]
        title = ff == fdom.coordinates[0]
        for pol in pdom.labels:
            axx = axs.pop(0)
            _plot_single_freq(
                    axx,
                    ift.makeField(sdom, sky_field.val[pdom.label2index(pol), 0, ii]),
                    title=f"Stokes {pol}" if title else "",
                    colorbar=colorbar,
                    offset=offset,
                    norm=LogNorm() if pol == "I" else None,
                    cmap="inferno" if pol == "I" else "seismic",
                    vmin=vmin[pol], vmax=vmax[pol])

        loop_fdom = IRGSpace([ff])
        loop_dom = pdom, tdom, loop_fdom, sdom
        loop_sky = ift.makeField(loop_dom, sky_field.val[:, :, ii:ii+1])
        ang = polarization_angle(loop_sky).val[0, 0]
        lin = linear_polarization(loop_sky).val[0, 0]

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
            plt.colorbar(im, orientation="horizontal", ax=axx, format=StrMethodFormatter("{x:.0f}°")).set_ticks([-90, -45, 0, 45, 90])


    plt.tight_layout()
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
        plt.close()


def _plot_single_freq(axx, field, title, colorbar=True, offset=None, **kwargs):
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


def polarization_quiver(ax, sky_field):
    assert_sky_domain(sky_field.domain)
    pdom, tdom, fdom, sdom = sky_field.domain
    assert all((pol in pdom.labels) for pol in ["I", "Q", "U"])
    assert tdom.size == fdom.size == 1
    ang = polarization_angle(sky_field).val[0, 0]
    lin = linear_polarization(sky_field).val[0, 0]

    im = ax.imshow(lin.T, cmap="inferno", norm=LogNorm(), origin="lower",
                   extent=_extent(sdom))
    scale = np.max(lin)*max(lin.shape) * 5

    nx, ny = sdom.shape
    Y, X = np.meshgrid(np.linspace(*ax.get_ylim(), ny, endpoint=True),
                       np.linspace(*ax.get_xlim(), nx, endpoint=True))
    ax.quiver(X, Y, -lin*np.sin(ang), lin*np.cos(ang),
              angles="uv", pivot='middle',
              headlength=0, headwidth=1, minlength=0, minshaft=.1, width=0.01,
              units="xy",
              scale_units="xy",
              scale=scale,
              alpha=.3,
              )


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
