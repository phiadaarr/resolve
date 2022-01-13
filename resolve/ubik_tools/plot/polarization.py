import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import nifty8 as ift
import numpy as np
from ...irg_space import IRGSpace

from ...util import assert_sky_domain


def polarization_overview(sky_field, name=None):
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
        for pol in pdom.labels:
            axx = axs.pop(0)
            _plot_single_freq(
                    axx,
                    ift.makeField(sdom, sky_field.val[pdom.label2index(pol), 0, ii]),
                    title=f"Stokes {pol}",
                    norm=LogNorm() if pol == "I" else None,
                    cmap="inferno" if pol == "I" else "seismic",
                    vmin=vmin[pol], vmax=vmax[pol])
        loop_fdom = IRGSpace([ff])
        loop_dom = pdom, tdom, loop_fdom, sdom
        _polarization_plot_detail(axs.pop(0), ift.makeField(loop_dom, sky_field.val[:, :, ii:ii+1]))

    plt.tight_layout()
    if name is None:
        plt.show()
    else:
        plt.savefig(name)
        plt.close()


def _plot_single_freq(axx, field, title, **kwargs):
    assert len(field.shape) == 2
    axx.imshow(field.val.T, extent=_extent(field.domain), origin="lower", **kwargs)
    axx.set_title(title)


def _extent(sdom):
    sdom = ift.DomainTuple.make(sdom)
    assert len(sdom) == 1
    sdom = sdom[0]
    nx, ny = sdom.shape
    dx, dy = sdom.distances
    xlim, ylim = nx*dx/2, ny*dy/2
    return [-xlim, xlim, -ylim, ylim]


def _polarization_plot_detail(ax, sky_field):
    assert_sky_domain(sky_field.domain)
    pdom, tdom, fdom, sdom = sky_field.domain
    assert all((pol in pdom.labels) for pol in ["I", "Q", "U"])
    assert tdom.size == fdom.size == 1
    val = {kk: sky_field.val[pdom.label2index(kk), 0, 0] for kk in pdom.labels}
    ang = polarization_angle(sky_field).val[0, 0]
    lin = linear_polarization(sky_field).val[0, 0]

    im = ax.imshow(lin.T, cmap="inferno", origin="lower", norm=LogNorm(),
                    extent=_extent(sdom))

    nx, ny = sdom.shape
    Y, X = np.meshgrid(np.linspace(*ax.get_xlim(), nx, endpoint=True),
                       np.linspace(*ax.get_ylim(), ny, endpoint=True))
    ax.quiver(X, Y, -lin*np.sin(ang), lin*np.cos(ang),
              headlength=0, headwidth=1, minlength=0, minshaft=1, width=0.01, units="x",
              angles="uv", pivot='middle', alpha=.3)


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
