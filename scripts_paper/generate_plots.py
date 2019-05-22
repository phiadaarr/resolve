import pickle
from os.path import join

import matplotlib as mpl
import numpy as np

import nifty5 as ift
import resolve as rve
from problem import Problem
from utils import (paper_plot_antenna, paper_plot_calib_solutions,
                   paper_plot_pspec, paper_plot_timeline, paper_plot_uv,
                   plot_amplitude_op, plot_integrated, plot_sky, plot_weighted)


def myplots(directory, cglimit, nsamps, position):
    plm = Problem(rve.ConfigurationParser(join(directory, 'config.txt')))
    dhs = plm.dhs
    cal_ops = plm.cal_ops
    rs = plm.rs
    diffuse = plm.diffuse
    lh = plm.lh

    position = rve.load_pickle(join(directory, position + '.pickle'))
    try:
        ground_truth = rve.load_pickle(join(directory, 'mock.pickle'))
    except FileNotFoundError:
        ground_truth = None

    fname = join(directory, 'samples_for_plotting.pickle')
    if nsamps > 1:
        print('Draw samples')
        ic = ift.GradientNormController(iteration_limit=cglimit)
        res_samples = rve.MetricGaussianKL(position, lh, ic, nsamps).samples
        with open(fname, 'wb') as f:
            pickle.dump(res_samples, f, pickle.HIGHEST_PROTOCOL)
    else:
        print('Load samples')
    res_samples = rve.load_pickle(fname)

    if ground_truth is not None:
        print('Plot histograms')
        # plot_integrated(position, directory + 'integrated0', diffuse,
        #                 ground_truth, res_samples, 35, 47, 35, 47)
        plot_integrated(position, directory + 'integrated', diffuse,
                        ground_truth, res_samples, 37, 48, 11, 22)

        tmin, tmax = 216, 230
        tmin, tmax = 0, 1e6
    else:
        tmin, tmax = 0, 1e6

    print('Plot timeline')
    paper_plot_timeline(directory + 'timeline', dhs['c'], 1, 3, 0, tmin, tmax)

    print('Plot power spectrum')
    paper_plot_pspec(position, directory + 'pspec', diffuse.pspec, res_samples,
                     ground_truth)

    print('Generate sky plots')
    plot_sky(rs['t'].adjoint(ift.makeOp(dhs['t'].var).inverse(dhs['t'].vis)),
             directory + 'j_science', False,
             r'$\mathrm{Jy}^{-1}\cdot\mathrm{rad}^{-2}$')
    sc = ift.StatCalculator()
    for samp in res_samples:
        sc.add(diffuse.force(position + samp))
    disable_axes = False
    cblabel = r'$\mathrm{Jy}\cdot\mathrm{rad}^{-2}$'
    dct = {}
    if ground_truth is not None:
        gt = diffuse.force(ground_truth)
        dct = {
            'vmin': gt.to_global_data().min(),
            'vmax': gt.to_global_data().max()
        }
        plot_sky(gt, directory + 'sky_groundtruth', disable_axes, cblabel,
                 **dct)
        res = abs(gt - sc.mean)
        plot_sky(res, directory + 'sky_residual', disable_axes, cblabel)

        res = gt - sc.mean
        plot_weighted(directory + 'weighted', res/sc.var.sqrt())

        n = res.domain.size
        onesig = np.sum((res < sc.var.sqrt()).to_global_data())/n
        twosig = np.sum((res < 2*sc.var.sqrt()).to_global_data())/n
        threesig = np.sum((res < 3*sc.var.sqrt()).to_global_data())/n
        print('Reconstruction with one sigma:', onesig)
        print('Reconstruction with two sigma:', twosig)
        print('Reconstruction with three sigma:', threesig)
    plot_sky(sc.mean, directory + 'sky_posteriormean', disable_axes, cblabel,
             **dct)
    plot_sky(
        sc.mean,
        directory + 'logsky_posteriormean',
        disable_axes,
        cblabel,
        log=True)
    plot_sky(sc.var.sqrt(), directory + 'sky_sd', disable_axes, cblabel)
    plot_sky(
        sc.var.sqrt()/sc.mean,
        directory + 'sky_rel_sd',
        disable_axes,
        '',
        log=True)

    print('Plot uv coverage')
    paper_plot_uv(directory + 'uv_coverage', dhs)

    print('Plot all calibration at once')
    paper_plot_calib_solutions(position, directory + 'comparison_calibration',
                               cal_ops, dhs['c'], ground_truth, res_samples)

    mode, ant, pol = 'ph', 15, '1'
    paper_plot_antenna(position, directory + 'example_phase',
                       cal_ops[mode + pol], dhs, res_samples, ant, mode,
                       ground_truth)

    mode, ant, pol = 'ampl', 0, '0'
    paper_plot_antenna(position, directory + 'example_amplitude',
                       cal_ops[mode + pol], dhs, res_samples, ant, mode,
                       ground_truth)


if __name__ == '__main__':
    np.seterr(all='raise', under='warn')
    ift.fft.enable_fftw()

    nice_fonts = {
        'text.usetex': True,
        'font.family': 'serif',
        'axes.labelsize': 8,
        'font.size': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
    }
    nice_fonts['text.latex.preamble'] = [
        r'\usepackage[varg]{txfonts}', r'\usepackage{nicefrac}'
    ]
    mpl.rcParams.update(nice_fonts)

    np.random.seed(42)
    plot_amplitude_op()

    np.random.seed(42)
    myplots('mock', 400, 100, '20_imaging')

    np.random.seed(42)
    myplots('sn', 400, 100, '32_imaging_adjvar')
