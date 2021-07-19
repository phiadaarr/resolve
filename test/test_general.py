# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

from os.path import join

import numpy as np
import pytest

import nifty8 as ift
import resolve as rve

pmp = pytest.mark.parametrize
np.seterr(all="raise")

direc = "/data/"
nthreads = 1
ift.set_nthreads(nthreads)
rve.set_nthreads(nthreads)
rve.set_epsilon(1e-4)
rve.set_wgridding(False)
OBS = []
for polmode in ["all", "stokesi", ["LL"], "stokesiavg"]:
    OBS.append(
        rve.ms2observations(
            f"{direc}CYG-ALL-2052-2MHZ.ms", "DATA", True, 0, polarizations=polmode
        )[0]
    )
assert OBS[1].max_snr() >= OBS[2].max_snr()  # Average data so SNR increases
npix, fov = 256, 1 * rve.DEG2RAD
dom = ift.RGSpace((npix, npix), (fov / npix, fov / npix))
sky0 = ift.SimpleCorrelatedField(
    dom, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5)
).exp()
inserter = rve.PointInserter(sky0.target, np.array([[0, 0]]))
points = ift.InverseGammaOperator(
    inserter.domain, alpha=0.5, q=0.2 / dom.scalar_dvol
).ducktape("points")
sky = rve.vla_beam(dom, np.mean(OBS[0].freq)) @ (sky0 + inserter @ points)

freqdomain = rve.IRGSpace(np.linspace(1000, 1050, num=10))
FACETS = [(1, 1), (2, 2), (2, 1), (1, 4)]


@pmp(
    "ms",
    ("CYG-ALL-2052-2MHZ", "CYG-D-6680-64CH-10S", "AM754_A030124_flagged", "1052735056"),
)
@pmp("with_calib_info", (False, True))
@pmp("compress", (False, True))
def test_save_and_load_observation(ms, with_calib_info, compress):
    ms = f"{direc}{ms}.ms"
    for spw in range(rve.ms_n_spectral_windows(ms)):
        obs = rve.ms2observations(ms, "DATA", with_calib_info, spectral_window=spw)
        for ob in obs:
            ob.save("foo.npz", compress)
            ob1 = rve.Observation.load("foo.npz")
            assert ob == ob1


@pmp("slc", [slice(3), slice(14, 15), slice(None, None, None), slice(None, None, 5)])
def test_sliced_readin(slc):
    ms = f"{direc}CYG-D-6680-64CH-10S.ms"
    obs0 = rve.ms2observations(ms, "DATA", False, 0)[0]
    obs = rve.ms2observations(ms, "DATA", False, 0, channel_slice=slc)[0]
    ch0 = obs0.weight.val[..., slc]
    ch = obs.weight.val
    assert ch0.ndim == 3
    assert ch0.shape == ch.shape
    np.testing.assert_equal(ch0, ch)


def try_operator(op):
    pos = ift.from_random(op.domain)
    op(pos)
    lin = op(ift.Linearization.make_var(pos))
    lin.gradient


def try_lh(obs, lh_class, *args):
    if obs.polarization.has_crosshanded():
        with pytest.raises(RuntimeError):
            lh_class(*args)
        return
    try_operator(lh_class(*args))


@pmp("obs", OBS)
def test_imaging_likelihood(obs):
    try_lh(obs, rve.ImagingLikelihood, obs, sky)


@pmp("obs", OBS)
def test_varcov_imaging_likelihood(obs):
    var = rve.divide_where_possible(1, obs.weight)
    invcovop = (
        ift.InverseGammaOperator(obs.vis.domain, 1, var).reciprocal().ducktape("invcov")
    )
    try_lh(obs, rve.ImagingLikelihood, obs, sky, invcovop)


@pmp("obs", OBS)
@pmp("noisemodel", range(2))
def test_weighting_methods(obs, noisemodel):
    efflen = obs.effective_uvwlen()
    npix = 2500
    dom = ift.RGSpace(npix, 2 * max(efflen) / npix)
    baseline_distributor = ift.LinearInterpolator(dom, efflen.T)
    pol_freq_copy = ift.ContractionOperator(obs.vis.domain, (0, 2)).adjoint
    cf = ift.SimpleCorrelatedField(
        dom, 0, (1, 1), (1, 1), (1.2, 0.4), (0.5, 0.2), (-3, 0.5), "invcov"
    ).exp()
    correction = pol_freq_copy @ baseline_distributor @ cf
    if noisemodel == 0:  # Multiplicative noise model
        var = rve.divide_where_possible(1, obs.weight)
        invcovop = ift.makeOp(obs.weight) @ correction ** (-2)
    elif noisemodel == 1:  # Additive noise model
        var = rve.divide_where_possible(1, obs.weight)
        invcovop = (ift.Adder(var) @ correction ** 2).reciprocal()
    try_lh(obs, rve.ImagingLikelihood, obs, sky, invcovop)


@pmp("time_mode", [True, False])
def test_calibration_likelihood(time_mode):
    obs = rve.ms2observations(
        f"{direc}AM754_A030124_flagged.ms",
        "DATA",
        True,
        spectral_window=0,
        polarizations="stokesi",
    )
    t0, _ = rve.tmin_tmax(*obs)
    obs = [oo.move_time(-t0) for oo in obs]
    uants = rve.unique_antennas(*obs)
    utimes = rve.unique_times(*obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uants)}

    if time_mode:
        tmin, tmax = rve.tmin_tmax(*obs)
        assert tmin == 0
        npix = 128
        time_domain = ift.RGSpace(npix, 2 * (tmax - tmin) / npix)
        time_dct = None
    else:
        time_dct = {aa: ii for ii, aa in enumerate(utimes)}
        time_domain = ift.UnstructuredDomain(len(utimes))
    nants = len(uants)
    # total_N = npol*nants*nfreqs
    npol, nfreq = obs[0].npol, obs[0].nfreq
    total_N = npol * nants * nfreq
    dom = [
        ift.UnstructuredDomain(npol),
        ift.UnstructuredDomain(len(uants)),
        time_domain,
        ift.UnstructuredDomain(nfreq),
    ]
    if time_mode:
        reshaper = rve.Reshaper([ift.UnstructuredDomain(total_N), time_domain], dom)
        dct = {"offset_mean": 0, "offset_std": (1, 0.5)}
        dct1 = {
            "fluctuations": (2.0, 1.0),
            "loglogavgslope": (-4.0, 1),
            "flexibility": (5, 2.0),
            "asperity": None,
        }
        cfm = ift.CorrelatedFieldMaker("calibration_phases", total_N=total_N)
        cfm.add_fluctuations(time_domain, **dct1)
        cfm.set_amplitude_total_offset(**dct)
        phase = reshaper @ cfm.finalize(0)
        dct = {
            "offset_mean": 0,
            "offset_std": (1e-3, 1e-6),
        }
        dct1 = {
            "fluctuations": (2.0, 1.0),
            "loglogavgslope": (-4.0, 1),
            "flexibility": (5, 2.0),
            "asperity": None,
        }
        cfm = ift.CorrelatedFieldMaker("calibration_logamplitudes", total_N=total_N)
        cfm.add_fluctuations(time_domain, **dct1)
        cfm.set_amplitude_total_offset(**dct)
        logampl = reshaper @ cfm.finalize(0)
    lh, constantshape = None, (obs[0].vis.shape[0], obs[0].vis.shape[2])
    for ii, oo in enumerate(obs):
        oo = obs.pop(0)
        assert constantshape == (oo.vis.shape[0], oo.vis.shape[2])
        if not time_mode:
            mean, std = 0, np.pi / 2
            phase = ift.Adder(mean, domain=dom) @ ift.ducktape(
                dom, None, "calibration_phases"
            ).scale(std)
            mean, std = 0, 1
            logampl = ift.Adder(mean, domain=dom) @ ift.ducktape(
                dom, None, "calibration_logamplitudes"
            ).scale(std)

        abc = rve.calibration_distribution(oo, phase, logampl, antenna_dct, time_dct)
        if ii in [1, 2]:
            model_visibilities = ift.full(oo.vis.domain, 1)
            op = rve.CalibrationLikelihood(oo, abc, model_visibilities)
        else:
            op = rve.ImagingLikelihood(oo, sky, calibration_operator=abc)
        lh = op if lh is None else lh + op
    try_operator(lh)


@pmp("dtype", [np.float64, np.complex128])
def test_simple_operator(dtype):
    op = rve.AddEmptyDimension(ift.UnstructuredDomain(10))
    ift.extra.check_linear_operator(op, dtype, dtype)


@pmp("obs", OBS)
def test_calibration_distributor(obs):
    tgt = obs.vis.domain
    utimes = rve.unique_times(obs)
    uants = obs.antenna_positions.unique_antennas()
    dom = [
        ift.UnstructuredDomain(nn)
        for nn in [obs.npol, len(uants), len(utimes), obs.nfreq]
    ]
    uants = rve.unique_antennas(obs)
    time_dct = {aa: ii for ii, aa in enumerate(utimes)}
    antenna_dct = {aa: ii for ii, aa in enumerate(uants)}
    op = rve.calibration.CalibrationDistributor(
        dom,
        tgt,
        obs.antenna_positions.ant1,
        obs.antenna_positions.time,
        antenna_dct,
        time_dct,
    )
    ift.extra.check_linear_operator(op)
    # FIXME Separate test for rve.calibration.MyLinearInterpolator()


def test_point_inserter():
    dom = ift.RGSpace([16, 16], [0.5, 2])
    op = rve.PointInserter(dom, [[0, 4], [0, 0]])
    ift.extra.check_linear_operator(op)
    res = op(ift.full(op.domain, 1)).val_rw()
    assert res[8, 8] == 1
    assert res[8, 10] == 1
    res[8, 10] = res[8, 8] = 0
    assert np.all(res == 0)


def test_integrator_values():
    # FIXME Write also test which tests first bin from explicit formula
    domain = ift.RGSpace((12, 12))
    a0 = ift.ScalingOperator(domain, 0.0).ducktape("a0")
    b0 = ift.ScalingOperator(domain, 0.0).ducktape("b0")
    intop = rve.WienerIntegrations(freqdomain, domain).ducktape("int")
    logsky = rve.IntWProcessInitialConditions(a0, b0, intop)
    pos = ift.from_random(logsky.domain)
    out = logsky(pos)
    np.testing.assert_equal(out.val[0], a0.force(pos).val)


def test_response_distributor():
    dom = ift.UnstructuredDomain(2), ift.UnstructuredDomain(4)
    op0 = ift.makeOp(ift.makeField(dom, np.arange(8).reshape(2, -1)))
    op1 = ift.makeOp(ift.makeField(dom, 2 * np.arange(8).reshape(2, -1)))
    op = rve.response.ResponseDistributor(op0, op1)
    ift.extra.check_linear_operator(op)


@pmp("obs", OBS)
@pmp("facets", FACETS)
def test_single_response(obs, facets):
    mask = obs.mask.val
    op = rve.response.SingleResponse(dom, obs.uvw, obs.freq, mask[0],
                                     facets=facets)
    ift.extra.check_linear_operator(op, np.float64, np.complex64,
                                    only_r_linear=True, rtol=1e-6, atol=1e-6)


def test_facet_consistency():
    obs = OBS[0]
    res0 = None
    pos = ift.from_random(dom)
    for facets in FACETS:
        op = rve.response.SingleResponse(dom, obs.uvw, obs.freq, obs.mask.val[0],
                                         facets=facets)
        res = op(pos)
        if res0 is None:
            res0 = res
        ift.extra.assert_allclose(res0, res, atol=1e-4, rtol=1e-4)


@rve.onlymaster
def fvalid():
    return 1.0


@rve.onlymaster
def finvalid():
    ift.from_random(ift.UnstructuredDomain(1))
    return 1.0


def test_randomonmaster():
    fvalid()
    with pytest.raises(RuntimeError):
        finvalid()


def test_mfweighting():
    nrow = 100
    nchan = 4
    effuv = ift.random.current_rng().random((nrow, nchan))
    dom = ift.UnstructuredDomain(nchan), ift.RGSpace(npix, 2 * np.max(effuv) / npix)
    op = rve.MfWeightingInterpolation(effuv, dom)
    ift.extra.check_linear_operator(op)


def test_mf_response():
    ms = join(direc, "CYG-D-6680-64CH-10S.ms")
    obs = rve.ms2observations(ms, "DATA", False, 0, "stokesiavg")[0]
    frequency_domain = rve.IRGSpace(obs.freq)
    R = rve.MfResponse(obs, frequency_domain, dom)
    ift.extra.check_linear_operator(
        R, rtol=1e-5, atol=1e-5, target_dtype=np.complex128, only_r_linear=True
    )


def test_intop():
    dom = ift.RGSpace((12, 12))
    op = rve.WienerIntegrations(freqdomain, dom)
    ift.extra.check_linear_operator(op)


def test_prefixer():
    op = rve.KeyPrefixer(
        ift.MultiDomain.make({"a": ift.UnstructuredDomain(10), "b": ift.RGSpace(190)}),
        "invcov_inp",
    ).adjoint
    ift.extra.check_linear_operator(op)
