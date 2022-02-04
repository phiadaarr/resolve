import resolve as rve
import numpy as np
import nifty8 as ift

rve.set_epsilon(1e-9)
rve.set_wgridding(False)
rve.set_nthreads(8)
obslist = rve.ms2observations_all("/data/CYG-ALL-2052-2MHZ.ms", "DATA")
#obslist = rve.ms2observations_all("/data/CYG-ALL-13360-8MHZ.ms", "DATA")
obs = next(obslist)

obs = obs.restrict_to_stokesi()
model_vis = 0.*obs.vis

npix = 1000
dst = rve.str2rad("0.05deg") / npix
domain = rve.default_sky_domain(sdom=ift.RGSpace((npix, npix), (dst, dst)))

# obs0 = obs.average_stokesi()[0:1]
#
# vis1 = np.repeat(obs0.vis.val, 2, axis=1)
# weight1 = np.repeat(obs0.weight.val, 2, axis=1)
# antpos1 = rve.AntennaPositions(np.repeat(obs0.uvw, 2, axis=0))
# obs1 = rve.Observation(antpos1, vis1, weight1, obs0.polarization, obs.freq, obs._auxiliary_tables)
# wgts0 = rve.uniform_weights(obs0, domain).val
# wgts1 = rve.uniform_weights(obs1, domain).val
#
# print(wgts0)
# print(wgts1)
#
# assert np.all(weight1 != 0)
# assert np.all(wgts0 != 0)
# assert np.all(wgts1 != 0)
#
# assert np.sum(wgts0) == np.sum(wgts1)

for major in range(5):
    dirty = rve.dirty_image(obs, "uniform", domain)
    psf   = rve.dirty_image(obs, "uniform", domain, vis=obs.vis*0.+1.)

    p = ift.Plot()
    p.add(dirty.ducktape_left(domain[-1]), title="dirty")
    p.add(psf.ducktape_left(domain[-1]), title="psf")
    p.output(name="debug.png")

    rve.ubik_tools.field2fits(dirty, f"dirty_{major}.fits", [obs])
    rve.ubik_tools.field2fits(psf, "psf.fits", [obs])

    #psf_norm = psf.ducktape_left(domain[-1]).integrate()

    fft = ift.HartleyOperator(domain, space=3)
    vis2dprime = lambda x: fft(rve.dirty_image(obs, "uniform", domain, vis=x))

    ones = obs.vis*0.+1.
    Rprime = ift.makeOp(vis2dprime(ones)   )
    dprime =            vis2dprime(obs.vis)
    t0     =            vis2dprime(model_vis)

    # Generate noise covariance in prime space
    sc = ift.StatCalculator()
    sig = 1/obs.weight.sqrt()
    for _ in range(10):
        # Draw noise in data space
        n = ift.from_random(obs.vis.domain, dtype=complex) * sig
        # Project into dataprime space
        sc.add(vis2dprime(n))
    foo = sc.var.val_rw()
    ind = foo < 1e-5*np.max(foo)
    foo[ind] = np.inf
    foo = 1/foo
    assert np.sum(foo == 0.) > 0
    Nprime_inv = ift.makeField(dprime.domain, foo)
    rve.ubik_tools.field2fits(Nprime_inv.ducktape_left(domain), "Ninv.fits", [obs])
    Nprime_inv = ift.makeOp(Nprime_inv)
    # /Generate noise covariance in prime space

    rve.ubik_tools.field2fits(dprime.ducktape_left(domain), "dprime.fits", [obs])
    rve.ubik_tools.field2fits(fft(psf).ducktape_left(domain), "Rprime.fits", [obs])

    #sky = ift.InverseGammaOperator(domain, alpha=0.5, q=0.2/domain[-1].scalar_dvol)
    sky = ift.ScalingOperator(domain, 1.).exp().scale(1e14)

    lh = ift.GaussianEnergy(mean=dprime, inverse_covariance=Nprime_inv) @ \
            ift.Adder(t0, neg=True) @ fft @ sky
    def callback(sl):
        s = ift.extra.minisanity(dprime, lambda x: Nprime_inv, fft @ sky, sl, False)
        print(s)
    _, pos = ift.optimize_kl(lh, 1, 0,
            ift.NewtonCG(ift.AbsDeltaEnergyController(1e-2, iteration_limit=20, convergence_level=3, name="newton")),
            None, None,
            plottable_operators={"sky": sky.log().ducktape_left(domain[3])},
            overwrite=True, inspect_callback=callback, return_final_position=True)

    rve.ubik_tools.field2fits(sky(pos), f"sky_{major}.fits", [obs])
    model_data = (rve.InterferometryResponse(obs, sky.target) @ sky)(pos)
    residual_data = obs.vis - model_data

    ssky = sky.ducktape_left(domain[-1])

    from matplotlib.colors import LogNorm
    p = ift.Plot()
    p.add(ssky(pos), norm=LogNorm(), title="Sky")
    p.output(name="loop.png")

    print("Model data")
    print(model_data)
    print()
    print("Observation vis")
    print(obs.vis)
    print()

    model_vis = model_data
    obs = rve.Observation(obs.antenna_positions, residual_data.val, obs.weight.val, obs.polarization,
            obs.freq, obs._auxiliary_tables)
