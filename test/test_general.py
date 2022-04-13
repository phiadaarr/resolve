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
# Copyright(C) 2020-2021 Max-Planck-Society
# Author: Philipp Arras

from os.path import join

import nifty8 as ift
import numpy as np
import pytest

import resolve as rve

from .common import setup_function, teardown_function

pmp = pytest.mark.parametrize
np.seterr(all="raise")

direc = "/data/"
OBS = []
for polmode in ["all", "stokesi", "stokesiavg"]:
    oo = rve.ms2observations(
            f"{direc}CYG-ALL-2052-2MHZ.ms", "DATA", True, 0, polarizations=polmode
        )[0]
    # OBS.append(oo.to_single_precision())
    OBS.append(oo.to_double_precision())
npix, fov = 256, 1 * rve.DEG2RAD
sdom = ift.RGSpace((npix, npix), (fov / npix, fov / npix))
sky = ift.SimpleCorrelatedField(sdom, 21, (1, 0.1), (5, 1), (1.2, 0.4), (0.2, 0.2), (-2, 0.5)).exp()
dom = rve.default_sky_domain(sdom=sdom, fdom=rve.IRGSpace([np.mean(OBS[0].freq)]))
sky = sky.ducktape_left(dom)

inserter = rve.PointInserter(dom, np.array([[0, 0]]))
points = ift.InverseGammaOperator(inserter.domain, alpha=0.5, q=0.2 / sdom.scalar_dvol).ducktape("points")
sky = sky + inserter @ points
sky = rve.vla_beam(dom) @ sky
rve.assert_sky_domain(sky.target)

freqdomain = rve.IRGSpace(np.linspace(1000, 1050, num=10))
FACETS = [(1, 1), (2, 2), (2, 1), (1, 4)]


@pmp("ms", ("CYG-ALL-2052-2MHZ", "CYG-D-6680-64CH-10S", "AM754_A030124_flagged"))
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


@pmp("slc", [slice(3), slice(14, 15), [14, 15], slice(None, None, None), slice(None, None, 5)])
def test_sliced_readin(slc):
    ms = f"{direc}CYG-D-6680-64CH-10S.ms"
    obs0 = rve.ms2observations(ms, "DATA", False, 0)[0]
    obs = rve.ms2observations(ms, "DATA", False, 0, channels=slc)[0]
    ch0 = obs0.weight.val[..., slc]
    ch = obs.weight.val
    assert ch0.ndim == 3
    assert ch0.shape == ch.shape
    np.testing.assert_equal(ch0, ch)


def test_legacy_load():
    rve.Observation.legacy_load(f"{direc}legacy.npz")


def test_flag_baseline():
    ms = f"{direc}CYG-D-6680-64CH-10S.ms"
    obs = rve.ms2observations(ms, "DATA", True, 0)[0]
    obs.flag_baseline(3, 5)


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
    obs = obs.restrict_to_stokesi()
    try_lh(obs, rve.ImagingLikelihood, obs, sky, 1e-6, False)


def test_imaging_likelihood_calibration_field():
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

    _, tmax = rve.tmin_tmax(*obs)
    antenna_dct = {aa: ii for ii, aa in enumerate(uants)}
    time_domain = ift.RGSpace(50, tmax / 50 * 2)
    nants = len(uants)
    npol, nfreq = obs[0].npol, obs[0].nfreq
    total_N = npol * nants * nfreq
    dom = [
        obs[0].polarization.space,
        ift.UnstructuredDomain(len(uants)),
        time_domain,
        ift.UnstructuredDomain(nfreq),
    ]
    mean, std = 0, np.pi / 2
    phase = ift.Adder(mean, domain=dom) @ ift.ducktape(
        dom, None, "calibration_phases"
    ).scale(std)
    mean, std = 0, 1
    logampl = ift.Adder(mean, domain=dom) @ ift.ducktape(
        dom, None, "calibration_logamplitudes"
    ).scale(std)
    abc = rve.calibration_distribution(obs[0], phase, logampl, antenna_dct)
    lh = rve.ImagingLikelihood(
        obs[0], sky, calibration_operator=abc, epsilon=1e-6, do_wgridding=False
    )
    try_operator(lh)
    pos = ift.from_random(lh.domain)
    calib_field = abc.force(pos)
    calib_field = rve.DtypeConverter(
        calib_field.domain, calib_field.dtype, obs[0].vis.dtype
    )(calib_field)
    lh1 = rve.ImagingLikelihood(
        obs[0], sky, calibration_field=calib_field, epsilon=1e-6, do_wgridding=False
    )
    ift.extra.assert_allclose(lh(pos), lh1.force(pos))


@pmp("obs", OBS)
def test_varcov_imaging_likelihood(obs):
    var = rve.divide_where_possible(1, obs.weight)
    invcovop = (
        ift.InverseGammaOperator(obs.vis.domain, 1, var).reciprocal().ducktape("invcov")
    )
    try_lh(obs, rve.ImagingLikelihood, obs, sky, 1e-6, False, invcovop.log(), None)


@pmp("obs", OBS)
@pmp("noisemodel", range(2))
def test_weighting_methods(obs, noisemodel):
    efflen = obs.effective_uvwlen().val[0]
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
    try_lh(obs, rve.ImagingLikelihood, obs, sky, 1e-6, False, invcovop, None)


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
        obs[0].polarization.space,
        ift.UnstructuredDomain(len(uants)),
        time_domain,
        ift.UnstructuredDomain(nfreq),
    ]
    if time_mode:
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
        phase = cfm.finalize(0).ducktape_left(dom)
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
        logampl = cfm.finalize(0).ducktape_left(dom)
    lh, constantshape = None, (obs[0].vis.shape[0], obs[0].vis.shape[2])
    for ii, oo in enumerate(obs):
        oo = obs.pop(0)
        assert constantshape == (oo.vis.shape[0], oo.vis.shape[2])
        if not time_mode:
            mean, std = 0, np.pi / 2
            phase = ift.Adder(mean, domain=dom) @ ift.ducktape(dom, None, "calibration_phases").scale(std)
            mean, std = 0, 1
            logampl = ift.Adder(mean, domain=dom) @ ift.ducktape(dom, None, "calibration_logamplitudes").scale(std)
        abc = rve.calibration_distribution(oo, phase, logampl, antenna_dct, time_dct)
        if ii in [1, 2]:
            model_visibilities = ift.full(oo.vis.domain, 1)
            op = rve.CalibrationLikelihood(oo, abc, model_visibilities)
        else:
            op = rve.ImagingLikelihood(oo, sky, calibration_operator=abc, epsilon=1e-6,
                                       do_wgridding=False)
        lh = op if lh is None else lh + op
    try_operator(lh)


@pmp("obs", OBS)
def test_calibration_distributor(obs):
    tgt = obs.vis.domain
    utimes = rve.unique_times(obs)
    uants = obs.antenna_positions.unique_antennas()
    dom = [obs.polarization.space] + \
        [ift.UnstructuredDomain(nn) for nn in [len(uants), len(utimes), obs.nfreq]]
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
    sdom = ift.RGSpace([16, 16], [0.5, 2])
    dom = rve.default_sky_domain(sdom=sdom)
    op = rve.PointInserter(dom, [[0, 4], [0, 0]])
    op.adjoint(ift.full(op.target, 1)).val_rw()
    return
    ift.extra.check_linear_operator(op)
    res = op(ift.full(op.domain, 1)).val_rw()
    assert np.all(res[..., 8, 8] == 1)
    assert np.all(res[8, 10] == 1)
    res[..., 8, 10] = res[..., 8, 8] = 0
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


def test_intop():
    dom = ift.RGSpace((12, 12))
    op = rve.WienerIntegrations(freqdomain, dom)
    ift.extra.check_linear_operator(op)
