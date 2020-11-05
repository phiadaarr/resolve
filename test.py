# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import numpy as np
import pytest

import nifty7 as ift
import resolve as rve

pmp = pytest.mark.parametrize
np.seterr(all='raise')

direc = '/data/'
nthreads = 1
ift.fft.set_nthreads(nthreads)
rve.set_nthreads(nthreads)
rve.set_epsilon(1e-4)
rve.set_wgridding(False)
OBSERVATION = rve.ms2observations(f'{direc}CYG-ALL-2052-2MHZ.ms', 'DATA', True, 0)[0]
snr = OBSERVATION.max_snr()
OBS = [OBSERVATION.restrict_to_stokes_i(), OBSERVATION.average_stokes_i()]
assert snr >= OBS[0].max_snr()  # Leave data out, so snr cannot increase
npix, fov = 256, 1*rve.DEG2RAD
dom = ift.RGSpace((npix, npix), (fov/npix, fov/npix))
sky0 = ift.SimpleCorrelatedField(dom, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5)).exp()
inserter = rve.PointInserter(sky0.target, np.array([[0, 0]]))
points = ift.InverseGammaOperator(inserter.domain, alpha=0.5, q=0.2/dom.scalar_dvol).ducktape('points')
sky = rve.vla_beam(dom, np.mean(OBSERVATION.freq)) @ (sky0 + inserter @ points)


@pmp('ms', ('CYG-ALL-2052-2MHZ', 'CYG-D-6680-64CH-10S', 'AM754_A030124_flagged'))
@pmp('with_calib_info', (False, True))
@pmp('compress', (False, True))
def test_save_and_load_observation(ms, with_calib_info, compress):
    spws = [0]
    if ms == 'AM754_A030124_flagged':
        spws.append(1)
    for spw in spws:
        obs = rve.ms2observations(f'{direc}{ms}.ms', 'DATA', with_calib_info, spectral_window=spw)
        for ob in obs:
            snr0 = ob.max_snr()
            print('Fraction useful:', ob.fraction_useful())
            ob.save_to_npz('foo.npz', compress)
            ob1 = rve.Observation.load_from_npz('foo.npz')
            assert ob == ob1
            assert snr0 <= ob1.max_snr()


def try_operator(op):
    pos = ift.from_random(op.domain)
    op(pos)
    lin = op(ift.Linearization.make_var(pos))
    lin.gradient


@pmp('obs', OBS)
def test_imaging_likelihood(obs):
    lh = rve.ImagingLikelihood(obs, sky)
    try_operator(lh)


@pmp('obs', OBS)
def test_varcov_likelihood(obs):
    var = rve.divide_where_possible(1, obs.weight)
    invcovop = ift.InverseGammaOperator(obs.vis.domain, 1, var).reciprocal().ducktape('invcov')
    lh = rve.ImagingLikelihoodVariableCovariance(obs, sky, invcovop)
    try_operator(lh)


@pmp('obs', OBS)
@pmp('noisemodel', range(2))
def test_imaging_likelihoods(obs, noisemodel):
    efflen = obs.effective_uvwlen()
    npix = 2500
    dom = ift.RGSpace(npix, 2*max(efflen)/npix)
    baseline_distributor = ift.LinearInterpolator(dom, efflen.T)
    pol_freq_copy = ift.ContractionOperator(obs.vis.domain, (0, 2)).adjoint
    cf = ift.SimpleCorrelatedField(dom, 0, (2, 2), (2, 2), (1.2, 0.4), (0.5, 0.2), (-2, 0.5), 'invcov').exp()
    correction = pol_freq_copy @ baseline_distributor @ cf
    if noisemodel == 0:  # Multiplicative noise model
        var = rve.divide_where_possible(1, obs.weight)
        invcovop = ift.makeOp(obs.weight) @ correction**(-2)
    elif noisemodel == 1:  # Additive noise model
        var = rve.divide_where_possible(1, obs.weight)
        invcovop = (ift.Adder(var) @ correction**2).reciprocal()
    lh = rve.ImagingLikelihoodVariableCovariance(obs, sky, invcovop)
    try_operator(lh)


@pmp('time_mode', [True, False])
def test_calibration_likelihood(time_mode):
    obs = rve.ms2observations(f'{direc}AM754_A030124_flagged.ms', 'DATA', True, spectral_window=0)
    obs = [oo.restrict_to_stokes_i() for oo in obs]
    t0, _ = rve.tmin_tmax(*obs)
    obs = [oo.move_time(-t0) for oo in obs]
    uants = rve.unique_antennas(*obs)
    utimes = rve.unique_times(*obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uants)}

    if time_mode:
        tmin, tmax = rve.tmin_tmax(*obs)
        assert tmin == 0
        npix = 128
        time_domain = ift.RGSpace(npix, 2*(tmax-tmin)/npix)
        time_dct = None
    else:
        time_dct = {aa: ii for ii, aa in enumerate(utimes)}
        time_domain = ift.UnstructuredDomain(len(utimes))
    nants = len(uants)
    # total_N = npol*nants*nfreqs
    npol, nfreq = obs[0].npol, obs[0].nfreq
    total_N = npol*nants*nfreq
    dom = [ift.UnstructuredDomain(npol), ift.UnstructuredDomain(len(uants)),
           time_domain, ift.UnstructuredDomain(nfreq)]
    if time_mode:
        reshaper = rve.Reshaper([ift.UnstructuredDomain(total_N), time_domain], dom)
        dct = {
            'offset_mean': 0,
            'offset_std': (1, 0.5),
            'prefix': 'calibration_phases'
        }
        dct1 = {
            'fluctuations': (2., 1.),
            'loglogavgslope': (-4., 1),
            'flexibility': (5, 2.),
            'asperity': None
        }
        cfm = ift.CorrelatedFieldMaker.make(**dct, total_N=total_N)
        cfm.add_fluctuations(time_domain, **dct1)
        phase = reshaper @ cfm.finalize(0)
        dct = {
            'offset_mean': 0,
            'offset_std': (1e-3, 1e-6),
            'prefix': 'calibration_logamplitudes'
        }
        dct1 = {
            'fluctuations': (2., 1.),
            'loglogavgslope': (-4., 1),
            'flexibility': (5, 2.),
            'asperity': None
        }
        cfm = ift.CorrelatedFieldMaker.make(**dct, total_N=total_N)
        cfm.add_fluctuations(time_domain, **dct1)
        logampl = reshaper @ cfm.finalize(0)
    lh, constantshape = None, (obs[0].vis.shape[0], obs[0].vis.shape[2])
    for ii, oo in enumerate(obs):
        oo = obs.pop(0)
        assert constantshape == (oo.vis.shape[0], oo.vis.shape[2])
        if not time_mode:
            mean, std = 0, np.pi/2
            phase = ift.Adder(mean, domain=dom) @ ift.ducktape(dom, None, 'calibration_phases').scale(std)
            mean, std = 0, 1
            logampl = ift.Adder(mean, domain=dom) @ ift.ducktape(dom, None, 'calibration_logamplitudes').scale(std)

        abc = rve.calibration_distribution(oo, phase, logampl, antenna_dct, time_dct)
        if ii in [1, 2]:
            model_visibilities = ift.full(oo.vis.domain, 1)
            op = rve.CalibrationLikelihood(oo, abc, model_visibilities)
        else:
            op = rve.ImagingCalibrationLikelihood(oo, sky, abc)
        lh = op if lh is None else lh + op
    try_operator(lh)


@pmp('dtype', [np.float64, np.complex128])
def test_simple_operator(dtype):
    op = rve.AddEmptyDimension(ift.UnstructuredDomain(10))
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp('obs', OBS)
def test_calibration_distributor(obs):
    tgt = obs.vis.domain
    utimes = rve.unique_times(obs)
    uants = obs.antenna_positions.unique_antennas()
    dom = [ift.UnstructuredDomain(nn) for nn in [obs.npol, len(uants), len(utimes), obs.nfreq]]
    uants = rve.unique_antennas(obs)
    time_dct = {aa: ii for ii, aa in enumerate(utimes)}
    antenna_dct = {aa: ii for ii, aa in enumerate(uants)}
    op = rve.calibration.CalibrationDistributor(dom, tgt, obs.antenna_positions.ant1, obs.antenna_positions.time, antenna_dct, time_dct)
    ift.extra.check_linear_operator(op)
    # TODO Separate test for rve.calibration.MyLinearInterpolator()


def test_point_inserter():
    dom = ift.RGSpace([16, 16], [0.5, 2])
    op = rve.PointInserter(dom, [[0, 4], [0, 0]])
    ift.extra.check_linear_operator(op)
    res = op(ift.full(op.domain, 1)).val_rw()
    assert res[8, 8] == 1
    assert res[8, 10] == 1
    res[8, 10] = res[8, 8] = 0
    assert np.all(res == 0)


def test_response_distributor():
    dom = ift.UnstructuredDomain(2), ift.UnstructuredDomain(4)
    op0 = ift.makeOp(ift.makeField(dom, np.arange(8).reshape(2, -1)))
    op1 = ift.makeOp(ift.makeField(dom, 2*np.arange(8).reshape(2, -1)))
    op = rve.response.ResponseDistributor(op0, op1)
    ift.extra.check_linear_operator(op)


@pmp('obs', OBS)
def test_single_response(obs):
    mask = obs.mask
    op = rve.response.SingleResponse(dom, obs.uvw, obs.freq, mask[0], False)
    ift.extra.check_linear_operator(op, np.float64, np.complex128, only_r_linear=True)
