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
from matplotlib.colors import LogNorm

import nifty5 as ift

from .calibration_distributor import CalibrationDistributor
from .likelihood import make_calibration
from .plot import Plot
from .sugar import tuple_to_image, zero_to_nan

c_sample = 'C0'
c_truth = 'C1'
c_mean = 'C2'


def plot_overview(position, name, diffuse, points, sky, cal_ops, dhs, rs):
    p = Plot()
    dct = {'colormap': 'inferno'}
    if diffuse:
        p.add(diffuse.force(position), title='Diffuse sky component', **dct)
        p.add(
            diffuse.pspec.force(position),
            title='Power spectrum of diffuse sky component',
            xlabel='Spatial frequency [1/rad]',
            **dct)
    if points:
        p.add(points.force(position), title='Point-like sky component', **dct)
    p.add(sky.force(position), title='Sky', **dct)
    p.add(sky.force(position), title='Sky', norm=LogNorm(), **dct)

    # Calibration fields
    dct = {
        'aspect': 'auto',
        'xlabel': 'Time [s]',
        'ylabel': 'Antenna',
    }
    phfunc = lambda x: tuple_to_image(180/np.pi*x.force(position))
    amplfunc = lambda x: tuple_to_image(x.exp().force(position))
    pspecs = {}
    dom = None
    for key, op in cal_ops.items():
        pspecs[key] = op.pspec.force(position)
        # if key[:-1] == 'ph':
        #     p.add(phfunc(op.nozeropad), title=key, **dct)
        # if key[:-1] == 'ampl':
        #     p.add(amplfunc(op.nozeropad), title=key, **dct)
        if key[:-1] == 'ph':
            p.add(phfunc(op), title=key, **dct)
        if key[:-1] == 'ampl':
            p.add(amplfunc(op), title=key, **dct)
    if len(cal_ops) > 0:
        dom = list(cal_ops.values())[0].target
        p.add(
            pspecs.values(),
            label=list(pspecs.keys()),
            xlabel='Frequency [1/s]',
            title='Power spectra of calibration solutions')

        for key, dh in dhs.items():
            tmp = lambda a: CalibrationDistributor(dom, a, dh.time)
            dtradj = (tmp(dh.ant1) + tmp(dh.ant2)).adjoint
            res = zero_to_nan(dtradj(ift.full(dtradj.domain, 1.)))
            p.add(
                tuple_to_image(res),
                title='Adjoint calibration distributor ({})'.format(key),
                **dct)

    # Dirty images
    for key, dh in dhs.items():
        nn = ift.makeOp(dh.var)
        jop = rs[key].adjoint @ nn.inverse
        p.add(jop(dh.vis), title='Dirty image {}'.format(key))
        if len(cal_ops) > 0:
            cop = make_calibration(dh, cal_ops).force(position)
            j = jop(dh.vis/cop)
            p.add(j, title='Dirty image {}, calibrated'.format(key))
    p.output(name='{}.png'.format(name), xsize=25, ysize=25)


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


def _pspec(position, samples, op, p, title, xlabel, ground_truth=None):
    sc = _stat(position, samples, op)
    pspecs, lw, labels, colors = [], [], [], []
    for ss in samples:
        pspecs.append(op.force(position + ss))
        lw.append(1)
        labels.append('')
        colors.append(c_sample)
    pspecs.append(sc.mean)
    lw.append(5)
    labels.append('Mean')
    colors.append(c_mean)
    if ground_truth is not None:
        pspecs.append(op.force(ground_truth))
        lw.append(4)
        colors.append(c_truth)
        labels.append('Ground truth')
    p.add(
        pspecs,
        title=title,
        xlabel=xlabel,
        linewidth=lw,
        label=labels,
        colors=colors)


def plot_sampled_overview(position,
                          name,
                          diffuse,
                          points,
                          sky,
                          cal_ops,
                          dhs,
                          rs,
                          samples=[],
                          ground_truth=None):
    p = Plot()

    if len(samples) > 1:
        sc = _stat(position, samples, sky)
        p.add(sc.mean, title='Sky', colormap='inferno')
        p.add(sc.var.sqrt(), title='Sky (Std deviation)')
        p.add(sc.var.sqrt()/sc.mean, title='Sky (Normalized std deviation)')
        p.add(
            sc.mean,
            title='Sky (log scale)',
            colormap='inferno',
            norm=LogNorm())
        if ground_truth is not None:
            for op, mysc in [(sky, sc)]:  #, (sky.log(), logsc)]:
                p.add(
                    op.force(ground_truth),
                    colormap='inferno',
                    title='Ground truth')
                p.add(
                    abs(op.force(ground_truth) - mysc.mean),
                    colormap='inferno',
                    title='|Ground truth-Mean|')
                p.add(
                    abs(op.force(ground_truth) - mysc.mean)/np.sqrt(mysc.var),
                    colormap='inferno',
                    title='|Ground truth-Mean|/Stddev',
                    zmax=3)
    if diffuse:
        title = 'Power spectrum of diffuse sky component'
        xlabel = 'Spatial frequency [1/rad]'
        _pspec(position, samples, diffuse.pspec, p, title, xlabel,
               ground_truth)

    if diffuse and points:
        if len(samples) > 1:
            sc = _stat(position, samples, diffuse)
            p.add(sc.mean, title='Diffuse sky component', colormap='inferno')
            p.add(sc.var.sqrt(), title='Diffuse sky component (Std deviation)')
            p.add(
                sc.var.sqrt()/sc.mean,
                title='Diffuse sky component (Normalized std deviation)')
            if ground_truth is not None:
                p.add(diffuse.force(ground_truth), title='Ground truth')
        if points:
            raise NotImplementedError

    # Calibration fields
    dct = {
        'aspect': 'auto',
        'xlabel': 'Time [s]',
        'ylabel': 'Antenna',
    }
    for key, op in cal_ops.items():
        if key[:-1] == 'ph':
            myop = 180/np.pi*op
        elif key[:-1] == 'ampl':
            myop = op.exp()
        else:
            raise RuntimeError
        sc = _stat(position, samples, myop, tuple_to_image)

        p.add(sc.mean, title=key, colormap='inferno', **dct)
        p.add(sc.var.sqrt(), title=key + ' (Std dev)', **dct)
        if ground_truth is not None:
            p.add(
                abs(tuple_to_image(myop.force(ground_truth)) - sc.mean)/
                np.sqrt(sc.var),
                colormap='inferno',
                title='|Ground truth-Mean|/Stddev',
                zmax=3,
                **dct)
        title = 'Power spectrum of calibration soluation ' + key
        xlabel = 'Frequency [1/s]'
        _pspec(position, samples, op.pspec, p, title, xlabel)

    if len(cal_ops) > 0:
        p.add(None)
        dom = list(cal_ops.values())[0].target
        for key, dh in dhs.items():
            tmp = lambda a: CalibrationDistributor(dom, a, dh.time)
            dtradj = (tmp(dh.ant1) + tmp(dh.ant2)).adjoint
            res = zero_to_nan(dtradj(ift.full(dtradj.domain, 1.)))
            p.add(
                res,
                title='Adjoint calibration distributor ({})'.format(key),
                **dct)

    p.output(name='{}.png'.format(name), xsize=25, ysize=25, nx=4)


def plot_antenna_examples(position,
                          name,
                          cal_ops,
                          dh,
                          samples,
                          antennas,
                          ground_truth=None):
    dom = list(cal_ops.values())[0].target
    tmp = lambda a: CalibrationDistributor(dom, a, dh.time)
    dtr = tmp(dh.ant1) + tmp(dh.ant2)
    calib_periods = (dtr.adjoint)(ift.full(dtr.target, 1.)).to_global_data_rw()
    calib_periods /= np.max(calib_periods)
    calib_periods[calib_periods == 0] = np.nan

    fig = plt.figure(figsize=[25, 25])

    nx, ny = len(antennas), 4

    i_plot = 1
    for key, op in cal_ops.items():
        t_space = op.target[1]
        npoints = t_space.shape[0]
        dist = t_space.distances[0]
        xcoord = np.arange(npoints, dtype=np.float64)*dist

        for ant in antennas:
            ax = fig.add_subplot(ny, nx, i_plot)
            i_plot += 1
            extr = ift.DomainTupleFieldInserter(op.target, 0, (ant,)).adjoint
            if key[:-1] == 'ph':
                myop = extr @ (180/np.pi*op)
                ax.set_ylabel('Phase [°]')
            elif key[:-1] == 'ampl':
                myop = extr @ op.exp()
                ax.set_ylabel('Amplitude')
            ax.set_title('Polarization {}, Antenna {}'.format(key, ant))
            ax.set_xlabel('Time [s]')
            sc = ift.StatCalculator()
            for ss in samples:
                fld = myop.force(position + ss)
                sc.add(fld)
                ax.plot(
                    xcoord,
                    fld.to_global_data(),
                    c_sample,
                    alpha=.3,
                    color=c_sample)
            ax.plot(
                xcoord,
                sc.mean.to_global_data(),
                linewidth=5,
                label='Mean',
                color=c_mean)
            if ground_truth is not None:
                ax.plot(
                    xcoord,
                    myop.force(ground_truth).to_global_data(),
                    color=c_truth,
                    linewidth=4,
                    label='Ground truth')
            plt.legend()

            x1, x2, y1, y2 = plt.axis()
            plt.imshow(
                calib_periods[ant:ant + 1],
                extent=(0, t_space.distances[0]*t_space.shape[0], y1, y2),
                aspect='auto',
                cmap='Greys',
                alpha=.2)
            plt.axis((x1, x2, y1, y2))
    fig.tight_layout()
    plt.savefig('{}.png'.format(name))
    plt.close()
