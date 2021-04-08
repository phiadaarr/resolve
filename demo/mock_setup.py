##################################################
# filament structure generator by free propatator in QM
# 1D (set mode=0), 2D (mode=1) or 3D (mode=2)
# 1. generate gaussian random field c0 and phi0
# 2. calculate wave function psi0 = exp(c0/2 - 1j*phi0/hbar)
# 3. initial density field rho0 = |psi0|^2 = exp(c0)
# 4. calculate wave function psi1 at time a by propagator
# propagator_h = np.exp(-1j * hbar * a / 2 * (k ** 2))
# 5. calculate density field at time a ( rho1 = |psi1|^2 )
# filament prior model :  rho1 = Rho1(xi)
# produce synthetic data from the filament prior model
# and then validate the filament prior model by MGVI
##################################################

import sys

import numpy as np

import nifty7 as ift


def main():
    # arg : mode(dimension)=1(2D), a(time), noise_amplitude
    if len(sys.argv) == 4:
        mode = int(sys.argv[1])
        a = float(sys.argv[2])
        noise_amplitude = float(sys.argv[3])
    else:
        mode = 1

    filename = "filament_synthetic_data_mode_{}_a_{}_noise_amp_{}".format(mode, a, noise_amplitude) + "{}.png"
    # Define 2d position space
    N_pixels = 256
    position_space = ift.RGSpace([N_pixels, N_pixels], distances=1 / N_pixels)

    ### Define harmonic space and harmonic transform

    harmonic_space = position_space.get_default_codomain()
    # HarmonicTransformOperator is actually Hartley transform, only take real value
    HT = ift.HarmonicTransformOperator(harmonic_space, position_space)
    # FFTOperator(real FFT) produce complex number, we need to take real number(by .real)
    ifft = ift.FFTOperator(harmonic_space, position_space)  # from k-space to position space
    fft = ifft.inverse


    ### 1.Generate gaussian random field c0 and phi0

    # random seed
    ift.random.push_sseq_from_seed(100)

    # c0 = gaussian random field, initial density (rho0) = exp(c0)
    # amplitude spectrum of c0(P ~ Amp^2 ~ k^-2)
    def amp_spec_c0(k):
        P0, k0, gamma = [0.00001, 1, 2]
        return np.sqrt(P0 / ((1. + (k / k0) ** 2) ** (gamma / 2)))

    # Operators(for Operators we use upper-case)
    C0h = ift.create_power_operator(harmonic_space, power_spectrum=amp_spec_c0).ducktape('c0h')
    C0 = HT(C0h)

    # phi0 = gaussian random field
    # amplitude spectrum(P~ Amp^2 ~ k^-6)
    def amp_spec(k):
        P0, k0, gamma = [0.01, 1, 6]
        return np.sqrt(P0 / ((1. + (k / k0) ** 2) ** (gamma / 2)))

    # Operators
    Phi0h = ift.create_power_operator(harmonic_space, power_spectrum=amp_spec).ducktape('phi0h')
    Phi0 = HT(Phi0h)


    ### 2.Calculate initial wave function operator Psi_0

    hbar = 5 * 10 ** -3
    # a = 1.0 # time scale
    Half_operator_ = ift.ScalingOperator(C0.target, 0.5)
    Hbar_operator = ift.ScalingOperator(Phi0.target, -1j / hbar)
    Complexifier = ift.Realizer(Phi0.target).adjoint
    Phase_operator = Hbar_operator @ Complexifier

    Half_operator = Half_operator_ @ Complexifier
    Psi_0 = ift.exp(Half_operator(C0) - Phase_operator(Phi0))
    #psi_0 = np.exp(c0 / 2 - 1j * phi0 / hbar)
    Psi_0h = fft(Psi_0)


    ### 3.Initial filament density operator Rho0
    Rho0 = ift.exp(C0)
    #Rho0 = ift.exp(Two_times_operator(ift.log(Psi_0)).real)


    ### 4.Wave function_1 at time a(by free propagator in QM)

    # length of k vector for each pixel
    k_values = harmonic_space.get_k_length_array()

    propagator_h = ift.exp(-1j * hbar * a / 2 * (k_values) ** 2) # to put noise, use k_values_n instead of k_values

    # propagator operator in harmonic space
    Propagator_h = ift.makeOp(propagator_h)

    # Wave function_1 operator(Psi_1)
    Psi_1h = Propagator_h(Psi_0h)  # psi_1h = propagator_h * psi_0h
    Psi_1 = ifft(Psi_1h)


    ### 5.Calculate filament density operator(Rho1)

    # absolute square operator for complex number
    conjOP = ift.ConjugationOperator(Psi_1._target)
    rls_psf = ift.Realizer(Psi_1.target)

    # filament density operator Rho1 at time a
    Rho1 = rls_psf @ (ift.ScalingOperator(Psi_1.target, 1) * conjOP) @ Psi_1

    ### Rho1 is the prior (rho1 = Rho1(xi))

    FILAMENT_PRIOR = Rho1



    ### MGVI part
    # produce synthetic data from the filament prior model
    # and then validate the model by MGVI

    # signal is the filament prior model
    signal = FILAMENT_PRIOR

    # Build the Identity response and define signal response
    R = ift.ScalingOperator(position_space, factor=1.)
    signal_response = R(signal)

    # Specify noise
    data_space = R.target
    # noise_amplitude = 1.0
    # noise_amplitude should be a bit larger than the order of magnitude your signal is in
    N = ift.ScalingOperator(data_space, noise_amplitude ** 2)

    # Generate mock signal and data
    mock_position = ift.from_random(signal_response.domain, 'normal')  # Multifield for c0, phi1
    data = signal_response(mock_position) + N.draw_sample_with_dtype(dtype=np.float64)

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(
        deltaE=0.05, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(
        name='Newton', deltaE=0.5, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)

    # Set up likelihood and information Hamiltonian (gaussian likelihood = P(d,s) = P(d|s)P(s) = G(d-Rs,N))
    likelihood = (ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @
                  signal_response)
    H = ift.StandardHamiltonian(likelihood, ic_sampling)

    initial_mean = ift.from_random(H.domain, 'normal') * 0.1
    mean = initial_mean

    plot = ift.Plot()
    plot.add(signal(mock_position), title='Ground Truth, a = {}'.format(a))
    plot.add(R.adjoint_times(data), title='Data, noise amplitude = {}'.format(noise_amplitude))
    plot.output(ny=1, nx=2, xsize=24, ysize=6, name=filename.format("setup"))

    # number of samples used to estimate the KL
    N_samples = 10

    # Draw new samples to approximate the KL five times
    for i in range(5):
        # Draw new samples and minimize KL
        KL = ift.MetricGaussianKL.make(mean, H, N_samples, True)
        KL, convergence = minimizer(KL)
        mean = KL.position
        ift.extra.minisanity(data, lambda x: N.inverse, signal_response,
                             KL.position, KL.samples)

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(signal(KL.position), title="Latent mean")
        plot.output(ny=1, ysize=6, xsize=16,
                    name=filename.format("loop_{:02d}".format(i)))

    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    plot.add(sc.mean, title="Posterior Mean")
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

    plot.output(ny=1, nx=2, xsize=24, ysize=6, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))


if __name__ == '__main__':
    main()