import resolve as rve
import numpy as np
import nifty8 as ift

rve.set_epsilon(1e-9)
rve.set_wgridding(False)
rve.set_nthreads(8)
obslist = rve.ms2observations_all("/data/CYG-ALL-2052-2MHZ.ms", "DATA")
obs = next(obslist)

obs = obs.restrict_to_stokesi()

npix = 2000
dst = rve.str2rad("0.05deg") / npix
domain = rve.default_sky_domain(sdom=ift.RGSpace((npix, npix), (dst, dst)))

for major in range(5):
    dirty = rve.dirty_image(obs, "uniform", domain)
    psf   = rve.dirty_image(obs, "uniform", domain, vis=obs.vis*0.+1.)

    rve.ubik_tools.field2fits(dirty, "dirty.fits", [obs])
    rve.ubik_tools.field2fits(psf, "psf.fits", [obs])

    fft = ift.HartleyOperator(domain, space=3)
    vis2dprime = lambda x: fft(rve.dirty_image(obs, "uniform", domain, vis=x))

    ones = obs.vis*0.+1.
    Rprime = ift.makeOp(vis2dprime(ones)   )
    dprime =            vis2dprime(obs.vis)

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

    sky = ift.InverseGammaOperator(domain, alpha=0.5, q=0.2/domain[-1].scalar_dvol)
    #sky = ift.ScalingOperator(domain, 1.).exp().scale(1e-10/domain[3].scalar_dvol)
    lh = ift.GaussianEnergy(mean=dprime, inverse_covariance=Nprime_inv) @ fft @ sky
    def callback(sl):
        s = ift.extra.minisanity(dprime, lambda x: Nprime_inv, fft @ sky, sl, False)
        print(s)
    _, pos = ift.optimize_kl(lh, 1, 0, ift.NewtonCG(ift.AbsDeltaEnergyController(1e-2, iteration_limit=5, name="newton")), None, None, plottable_operators={"sky": sky.log().ducktape_left(domain[3])}, overwrite=True, inspect_callback=callback, return_final_position=True)

    rve.ubik_tools.field2fits(sky(pos), f"sky_{major}.fits", [obs])
    model_data = (rve.InterferometryResponse(obs, sky.target) @ sky)(pos)
    residual_data = obs.vis - model_data

    obs = rve.Observation(obs.antenna_positions, residual_data.val, obs.weight.val, obs.polarization,
            obs.freq, obs._auxiliary_tables)
