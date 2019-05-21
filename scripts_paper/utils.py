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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

import nifty5 as ift
import resolve as rve

c_sample = 'C0'
c_truth = 'C1'
c_mean = 'C2'

savefig_cfg = {'format': 'png', 'bbox_inches': 'tight', 'dpi': 300}

showthecolumnwidth = 256.0748
showthetextwidth = 523.5307
maxsamples = 50


def set_size(fig_width_pt=showthecolumnwidth):
    inches_per_pt = 1/72.27
    golden_ratio = (5**.5 - 1)/2
    fig_width_in = fig_width_pt*inches_per_pt
    fig_height_in = fig_width_in*golden_ratio
    return (fig_width_in, fig_height_in)


def set_quad_size(fig_width_pt=showthecolumnwidth):
    inches_per_pt = 1/72.27
    fig_width_in = fig_width_pt*inches_per_pt
    fig_height_in = fig_width_in
    return (fig_width_in, fig_height_in)


def paper_plot_antenna(position,
                       name,
                       op,
                       dhs,
                       samples,
                       antenna,
                       mode,
                       ground_truth=None):
    dom = op.target
    dh = dhs['c']
    tmp = lambda a: rve.CalibrationDistributor(dom, a, dh.time)
    dtr = tmp(dh.ant1) + tmp(dh.ant2)
    calib_periods = (dtr.adjoint)(ift.full(dtr.target, 1.)).to_global_data_rw()

    t_space = op.target[1]
    npoints = t_space.shape[0]
    dist = t_space.distances[0]
    xscaling = 60*60
    xcoord = np.arange(npoints, dtype=np.float64)*dist/xscaling

    fig, axes = plt.subplots(
        figsize=set_size(),
        nrows=2,
        sharex=True,
        gridspec_kw={'height_ratios': [4, 1]})
    ax = axes[0]
    extr = ift.DomainTupleFieldInserter(op.target, 0, (antenna,)).adjoint
    if mode == 'ph':
        myop = extr @ (180/np.pi*op)
    elif mode == 'ampl':
        myop = extr @ op.exp()
    sc = ift.StatCalculator()
    for ii in range(min([len(samples), maxsamples])):
        fld = myop.force(position + samples[ii])
        sc.add(fld)
        ax.plot(
            xcoord,
            fld.to_global_data(),
            c_sample,
            alpha=.3,
            color=c_sample,
            linewidth=0.5)
    ax.plot(
        xcoord,
        sc.mean.to_global_data(),
        linewidth=1,
        label='Mean',
        color=c_mean)
    if ground_truth is not None:
        ax.plot(
            xcoord,
            myop.force(ground_truth).to_global_data(),
            color=c_truth,
            linewidth=1,
            label='Ground truth')

    x1, x2, y1, y2 = ax.axis()
    im = ax.imshow(
        calib_periods[antenna:antenna + 1]/t_space.distances[0],
        extent=(0, t_space.distances[0]*t_space.shape[0]/xscaling, y1, y2),
        aspect='auto',
        cmap='Greys',
        vmin=0,
        alpha=.2)
    ax.axis((x1, x2, y1, y2))
    cbar = fig.colorbar(im, ax=axes)
    cbar.ax.set_ylabel(
        r'Calibration data density [$\nicefrac{1}{s}$]',
        rotation=270,
        labelpad=10)

    ax = axes[1]
    if ground_truth is not None:
        ycoord = (sc.mean - myop.force(ground_truth)).to_global_data()
        ax.plot(xcoord, ycoord, color='k', linewidth=1, label='Ground truth')
    ax.axhline(0, linestyle='--', color='k', linewidth=0.5)

    sd = sc.var.sqrt().to_global_data()
    ax.fill_between(xcoord, -sd, sd, alpha=0.2, facecolor=c_sample)

    if ground_truth is not None:
        lim = np.max(np.abs(ycoord))
        ax.set_ylim(-lim, lim)
    ax.set_xlim(0, np.max(xcoord))

    ax.set_xlabel('Time [h]')

    plt.savefig('{}.{}'.format(name, savefig_cfg['format']), **savefig_cfg)
    plt.close()


def _stat(position, samples, op, post_op=None):
    assert len(samples) > 1
    sc = ift.StatCalculator()
    for ss in samples:
        r = op.force(position + ss)
        if post_op is None:
            sc.add(r)
        else:
            sc.add(post_op(r))
    return sc


def paper_plot_calib_solutions(position, name, cal_ops, dh, ground_truth,
                               samples):
    if ground_truth is None:
        figsize = set_size()
        nx = 2
    else:
        figsize = set_size(showthetextwidth)
        nx = 4
    figsize = set_size(showthetextwidth)
    fig, axs = plt.subplots(
        nrows=4, ncols=nx, sharex=True, sharey=True, figsize=figsize)
    irow = -1
    for mode in ['ampl', 'ph']:
        for pol in ['0', '1']:
            irow += 1
            cal_op = cal_ops[mode + pol]

            t_space = cal_op.target[1]
            xscaling = 60*60
            x1, x2, y1, y2 = plt.axis()

            dct = {'aspect': 'auto', 'origin': 'lower', 'cmap': 'inferno'}
            dct['extent'] = (0, t_space.distances[0]*t_space.shape[0]/xscaling,
                             y1, cal_op.target[0].shape[0])

            if mode == 'ph':
                myop = 180/np.pi*cal_op
            else:
                myop = cal_op.exp()
            sc = _stat(position, samples, myop)

            images = []
            icol = 0
            if ground_truth is not None:
                images.append(axs[irow, icol].imshow(
                    myop.force(ground_truth).to_global_data(), **dct))
                icol += 1
            images.append(axs[irow, icol].imshow(sc.mean.to_global_data(),
                                                 **dct))
            icol += 1
            maxcol = icol
            cb = fig.colorbar(images[-1], ax=axs[irow, 0:maxcol])

            tick_locator = ticker.MaxNLocator(nbins=3)
            cb.locator = tick_locator
            cb.update_ticks()

            images = []
            dct['vmin'] = 0
            if ground_truth is not None:
                images.append(axs[irow, icol].imshow(
                    (myop.force(ground_truth) - sc.mean).to_global_data(),
                    **dct))
                icol += 1
            images.append(axs[irow, icol].imshow(
                sc.var.sqrt().to_global_data(), **dct))
            icol += 1
            cb = fig.colorbar(images[-1], ax=axs[irow, maxcol:icol])

            tick_locator = ticker.MaxNLocator(nbins=3)
            cb.locator = tick_locator
            cb.update_ticks()

    for ii in range(4):
        axs[ii, 0].set_ylabel('Antenna')
    for ii in range(nx):
        axs[-1, ii].set_xlabel('Time [h]')
    icol = 0
    if ground_truth is not None:
        axs[0, icol].set_title('Ground Truth')
        icol += 1
    axs[0, icol].set_title('Posterior Mean')
    icol += 1
    if ground_truth is not None:
        axs[0, icol].set_title('Residual')
        icol += 1
    axs[0, icol].set_title('Posterior Std. Dev.')
    plt.savefig('{}.{}'.format(name, savefig_cfg['format']), **savefig_cfg)
    plt.close()


def paper_plot_uv(name, dhs):
    lim = max([np.max(abs(dhs['c'].uv)), np.max(abs(dhs['t'].uv))])
    plt.figure(figsize=set_quad_size())
    plt.plot(*dhs['c'].uv.reshape(-1, 2).T, '.', color='grey', markersize=0.2)
    plt.plot(
        *dhs['t'].uv.reshape(-1, 2).T, '.', color='tomato', markersize=0.2)
    plt.xlabel(r'$u [\lambda]$')
    plt.ylabel(r'$v [\lambda]$')
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    plt.savefig('{}.{}'.format(name, savefig_cfg['format']), **savefig_cfg)
    plt.close()


def plot_sky(f, name, disable_axes, cblabel, vmin=None, vmax=None, log=False):
    plt.figure(figsize=set_size())
    dom = f.domain[0]
    nx, ny = dom.shape
    dx, dy = dom.distances
    if log:
        from matplotlib.colors import LogNorm
        norm = LogNorm()
    else:
        norm = None
    im = plt.imshow(
        np.rot90(f.to_global_data()),
        extent=[0, nx*dx, 0, ny*dy],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        cmap='inferno')
    if disable_axes:
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
    else:
        plt.xlabel('Sky coordinate [rad]')
        plt.ylabel('Sky coordinate [rad]')
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel(cblabel, rotation=270, labelpad=10)
    plt.savefig('{}.{}'.format(name, savefig_cfg['format']), **savefig_cfg)
    plt.close()


def paper_plot_pspec(position, name, op, res_samples, ground_truth):
    sc = _stat(position, res_samples, op)
    xcoord = op.target[0].k_lengths
    plt.figure(figsize=set_size())
    for ii in range(min([maxsamples, len(res_samples)])):
        plt.plot(
            xcoord,
            op.force(position + res_samples[ii]).to_global_data(),
            color=c_sample,
            alpha=0.3,
            linewidth=0.5)
    plt.plot(xcoord, sc.mean.to_global_data(), color=c_mean, linewidth=1)
    if ground_truth is not None:
        from power_spectrum import pspec
        plt.plot(xcoord, pspec(xcoord), color=c_truth, linewidth=1)
    plt.xlabel('Spatial frequency [1/rad]')
    plt.xlim([xcoord[1], np.max(xcoord)])
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel(r'Power $[1]$')
    plt.savefig('{}.{}'.format(name, savefig_cfg['format']), **savefig_cfg)
    plt.close()


def paper_plot_timeline(name, dh, ant1, ant2, pol, tmin, tmax):
    plt.figure(figsize=set_size())
    xscaling = 60

    flag0 = np.logical_and(dh.ant1 == ant1, dh.ant2 == ant2)
    flag1 = np.logical_and(dh.ant1 == ant2, dh.ant2 == ant1)
    flag = np.logical_or(flag0, flag1)
    vis = dh.vis.val[flag][:, 0, pol]
    var = dh.var.val[flag][:, 0, pol]
    t = dh.time[flag]/xscaling
    std_abs = np.sqrt(var)

    flag = np.logical_and(tmin < t, t < tmax)
    vis = vis[flag]
    std_abs = std_abs[flag]
    t = t[flag]

    color_cycle = plt.gca()._get_lines.prop_cycler
    [next(color_cycle) for _ in range(3)]
    plt.errorbar(
        t,
        vis.real,
        yerr=std_abs,
        markersize=1,
        elinewidth=0.5,
        fmt='o',
        label='Real part')
    plt.errorbar(
        t,
        vis.imag,
        yerr=std_abs,
        markersize=1,
        elinewidth=0.5,
        fmt='o',
        label='Imaginary part')
    plt.legend()
    plt.xlabel('Time [min]')
    plt.ylabel('Visibility [Jy]')
    plt.savefig('{}.{}'.format(name, savefig_cfg['format']), **savefig_cfg)
    plt.close()


def plot_integrated(position, name, diffuse, ground_truth, samples, xmi, xma,
                    ymi, yma):
    mask = ift.full(diffuse.target, 0.).to_global_data_rw()
    mask[xmi:xma, ymi:yma] = 1
    mask = ift.from_global_data(diffuse.target, mask)

    f = lambda pos: (ift.makeOp(mask) @ diffuse).force(pos).integrate()
    vols = np.array([f(position + s) for s in samples])

    plt.figure(figsize=set_size())
    plt.axvline(x=f(ground_truth), color=c_truth)
    plt.hist(vols, bins=30, color=c_sample, density=True)

    plt.xlabel('Integrated flux [Jy]')
    plt.ylabel(r'Probability density [$\mathrm{Jy}^{-1}$]')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    inset_axes = inset_axes(plt.gca(), width="30%", height="30%", loc=1)
    fld = (mask*diffuse.force(ground_truth)).to_global_data_rw()
    plt.imshow(fld.T, origin="lower", cmap='inferno')
    plt.xticks([])
    plt.yticks([])

    plt.savefig('{}.{}'.format(name, savefig_cfg['format']), **savefig_cfg)
    plt.close()


def plot_amplitude_op():
    a, k0 = 4., 1.
    slm, ym = -2.5, 10

    lsp = ift.LogRGSpace(128, (0.1,), (-2.,))

    qht = ift.QHTOperator(lsp)
    sym = ift.SymmetrizingOperator(lsp)
    dom = qht.domain[0]

    # Make ceps op
    dim = len(dom.shape)
    shape = dom.shape
    q_array = dom.get_k_array()
    no_zero_modes = (slice(1, None),)*dim
    ks = q_array[(slice(None),) + no_zero_modes]
    cepstrum_field = np.zeros(shape)
    cepstrum_field[no_zero_modes] = (a/(1 + np.sum(
        (ks.T/k0)**2, axis=-1).T))**2
    for i in range(dim):
        fst_dims = (slice(None),)*i
        sl = fst_dims + (slice(1, None),)
        sl2 = fst_dims + (0,)
        cepstrum_field[sl2] = np.sum(cepstrum_field[sl], axis=i)
    cepstrum = ift.from_global_data(dom, cepstrum_field)
    sm = ift.makeOp(cepstrum)

    cf = ift.from_random('normal', dom)
    cf = (qht @ sm)(cf)

    symcf = sym(cf)

    sl = ift.SlopeOperator(lsp)
    slsp = sl.domain[0]
    slparams = ift.from_global_data(slsp, np.array([slm, ym]))
    slope = sl(slparams)
    slop = symcf + slope

    fig, axes = plt.subplots(2, 2, figsize=set_size())
    arr = np.concatenate([
        cf.exp().to_global_data()[1:],
        symcf.exp().to_global_data()[1:],
        symcf.exp().to_global_data()[1:],
        slop.exp().to_global_data()[1:]
    ])

    ylims = [np.min(arr)/2., np.max(arr)*2.]
    field = cf.exp()
    dom = field.domain[0]
    npoints = dom.shape[0]
    xcoord = dom.t_0 + np.arange(npoints - 1)*dom.bindistances[0]
    xcoord = np.exp(xcoord)
    ycoord = field.to_global_data()[1:]
    axes[0, 0].plot(xcoord, ycoord)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_ylim(ylims)
    axes[0, 0].set_xlim([np.min(xcoord), np.max(xcoord)])
    axes[0, 0].xaxis.set_visible(False)

    field = symcf.exp()
    dom = field.domain[0]
    npoints = dom.shape[0]
    xcoord = dom.t_0 + np.arange(npoints - 1)*dom.bindistances[0]
    xcoord = np.exp(xcoord)
    ycoord = field.to_global_data()[1:]
    axes[1, 0].plot(xcoord, ycoord)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_ylim(ylims)
    axes[1, 0].set_xlim([np.min(xcoord), np.max(xcoord)])
    axes[1, 0].set_xlabel(r'$|k|$')

    field = symcf.exp()
    dom = field.domain[0]
    npoints = dom.shape[0]
    xcoord = dom.t_0 + np.arange(npoints - 1)*dom.bindistances[0]
    xcoord = xcoord[:(npoints//2)]
    xcoord = np.exp(xcoord)
    ycoord = field.to_global_data()[1:][:(npoints//2)]
    axes[0, 1].plot(xcoord, ycoord)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_ylim(ylims)
    axes[0, 1].set_xlim([np.min(xcoord), np.max(xcoord)])
    axes[0, 1].xaxis.set_visible(False)
    axes[0, 1].yaxis.set_visible(False)

    field = slop.exp()
    dom = field.domain[0]
    npoints = dom.shape[0]
    xcoord = dom.t_0 + np.arange(npoints - 1)*dom.bindistances[0]
    xcoord = np.exp(xcoord[:(npoints//2)])
    ycoord = field.to_global_data()[1:][:(npoints//2)]
    axes[1, 1].plot(xcoord, ycoord, zorder=1)
    ycoord = slope.exp().to_global_data()[1:][:(npoints//2)]
    axes[1, 1].plot(xcoord, ycoord, zorder=0)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_ylim(ylims)
    axes[1, 1].yaxis.set_visible(False)
    axes[1, 1].set_xlim([np.min(xcoord), np.max(xcoord)])
    axes[1, 1].set_xlabel(r'$|k|$')

    axes[1, 0].minorticks_off()
    axes[1, 1].minorticks_off()

    plt.savefig('{}.{}'.format('amplitude_operator', savefig_cfg['format']),
                **savefig_cfg)
    plt.close()
