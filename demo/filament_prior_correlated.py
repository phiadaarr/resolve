##################################################
# filament structure generator by correlated field model
# 1D (set mode=0), 2D (mode=1) or 3D (mode=2)
# 1. generate c0 and phi0 by correlated field model
# 2. calculate wave function psi0 = exp(c0/2 - j*phi0/hbar)
# 3. initial density field rho0 = |psi0|^2
# 4. calculate wave function psi1 at time a by propagator
# propagator_h = np.exp(-1j * hbar * a / 2 * (k ** 2))
# 5. calculate density field rho1 at time a = |psi1|^2
# filament prior model :  rho1 = Rho1(xi)
# we can use different propagator to generate different pattern
# modified propagator_h = np.exp(-1j * hbar * a / 2 * (k ** exponent_k))
# use different exponent_k(e.g. exponent_k=4)
# put noise to k (k = k + n_k)
##################################################

import sys

import numpy as np

import nifty7 as ift

import matplotlib.pyplot as plt

import matplotlib.colors as colors

import matplotlib.cm as cm

import resolve as rve


def main():
    # arg : a(time)
    if len(sys.argv) == 2:
        a = float(sys.argv[1])

    else:
        a = 0.1


    # Define 2d position space
    N_pixels = 2000
    position_space = ift.RGSpace([N_pixels, N_pixels], distances=1 / N_pixels)


    ### Define harmonic space and harmonic transform

    harmonic_space = position_space.get_default_codomain()
    # HarmonicTransformOperator is actually Hartley transform, only take real value
    HT = ift.HarmonicTransformOperator(harmonic_space, position_space)
    # FFTOperator(real FFT) produce complex number, we need to take real number(by .real)
    ifft = ift.FFTOperator(harmonic_space, position_space)  # from k-space to position space
    fft = ifft.inverse


    ### Define meshgrid
    # N : Number of pixels, T = sample spacing(T = 1/N)
    N = N_pixels
    T = 1 / N_pixels
    x = np.linspace(0, N * T, N, endpoint=False)
    y = np.linspace(0, N * T, N, endpoint=False)
    # Produce 2d array
    [X, Y] = np.meshgrid(x, y)

    ### 1.Generate c0 and phi0 by correlated field model

    # random seed
    ift.random.push_sseq_from_seed(80) # 100


    # c0 by correlated field model, initial density (rho0) = exp(c0)
    cfmaker_c0 = ift.CorrelatedFieldMaker('')
    # add fluctuations, flexibility, asperity, loglogavgslope
    cfmaker_c0.add_fluctuations(position_space, (5.0, 1.0), (2.4, 0.8), None, (-4., 1.0), 'c0')
    cfmaker_c0.set_amplitude_total_offset(21., (1, 0.1))
    Correlated_field_c0 = cfmaker_c0.finalize()
    C0 = Correlated_field_c0

    # phi0 by correlated field model
    cfmaker_phi0 = ift.CorrelatedFieldMaker('')
    # add fluctuations, flexibility, asperity, loglogavgslope
    cfmaker_phi0.add_fluctuations(position_space, (0.1, 0.05), (1.0, 0.2), None, (-6., 1.0), 'phi0')
    cfmaker_phi0.set_amplitude_total_offset(0., (1.0, 0.1))
    Correlated_field_phi0 = cfmaker_phi0.finalize()
    Phi0 = Correlated_field_phi0

    ### 2.Calculate initial wave function operator Psi_0

    hbar = 5 * 10 ** -3
    # a = 1.0 # time scale
    Half_operator_ = ift.ScalingOperator(C0.target, 0.5)
    Hbar_operator = ift.ScalingOperator(Phi0.target, -1j / hbar)
    Complexifier = ift.Realizer(Phi0.target).adjoint
    Phase_operator = Hbar_operator @ Complexifier

    Half_operator = Half_operator_ @ Complexifier
    Psi_0 = ift.exp(Half_operator(C0) - Phase_operator(Phi0))
    # psi_0 = np.exp(c0 / 2 - 1j * phi0 / hbar)
    Psi_0h = fft(Psi_0)

    ### 3.Initial filament density operator Rho0
    Rho0 = ift.exp(C0)
    # Rho0 = ift.exp(Two_times_operator(ift.log(Psi_0)).real)

    ### 4.Wave function_1 at time a(by free propagator in QM)

    # length of k vector for each pixel
    k_values = harmonic_space.get_k_length_array()

    ### use gaussian distribution for k_values
    # Define meshgrid
    # N : Number of pixels
    N_ = N_pixels
    x_ = np.linspace(0, N_, N_, endpoint=False)
    y_ = np.linspace(0, N_, N_, endpoint=False)
    # Produce 2d array
    [X_, Y_] = np.meshgrid(x_, y_)

    propagator_h = ift.exp(-1j * hbar * a / 2 * (k_values) ** 2)  # to put noise, use k_values_n instead of k_values

    # propagator operator in harmonic space
    Propagator_h = ift.makeOp(propagator_h)

    # Wave function_1 operator(Psi_1)
    Psi_1h = Propagator_h(Psi_0h)  # psi_1h = propagator_h * psi_0h
    Psi_1 = ifft(Psi_1h)

    ### 5.Calculate filament density operator(Rho1)

    # absolute square operator for complex number
    conjOP = ift.ConjugationOperator(Psi_1._target)
    rls_psf = ift.Realizer(Psi_1.target)

    ### Rho1 is the filament prior (rho1 = Rho1(xi))
    Rho1 = rls_psf @ (ift.ScalingOperator(Psi_1.target, 1) * conjOP) @ Psi_1


    ### Generate field from prior operator

    mock_position = ift.from_random(Rho1.domain, 'normal')  # Multifield for c0, phi1

    c0 = C0.force(mock_position)
    rho0 = Rho0.force(mock_position)
    phi0 = Phi0.force(mock_position)
    rho1 = Rho1.force(mock_position)



    ### Plotting
    filename = "filament_prior_correlated_log_scale_a_{}_N_{}.png".format(a, N_pixels)
    fig, ((ax1, ax3, ax2)) = plt.subplots(1, 3, figsize=(15,5))

    #density0 = ax1.pcolormesh(X, Y, rho0.val, cmap=cm.inferno)  # linear scale
    density0 = ax1.pcolormesh(X, Y, rho0.val, norm=colors.LogNorm(vmin=rho0.val.min(), vmax=rho0.val.max()), cmap=cm.inferno) # log scale
    ax1.set_aspect('equal')
    ax1.set_title('Initial Density', fontsize=15)
    fig.colorbar(density0, ax=ax1)

    #density1 = ax2.pcolormesh(X, Y, rho1.val, cmap=cm.inferno)  # linear scale
    density1 = ax2.pcolormesh(X, Y, rho1.val, norm=colors.LogNorm(vmin=rho1.val.min(), vmax=rho1.val.max()), cmap=cm.inferno) # log scale
    ax2.set_aspect('equal')
    ax2.set_title('Filament Density, a = {}'.format(a), fontsize=15)
    fig.colorbar(density1, ax=ax2)

    phase = ax3.pcolormesh(X, Y, phi0.val, cmap=cm.gray)
    ax3.set_aspect('equal')
    ax3.set_title('Initial Phase', fontsize=15)
    fig.colorbar(phase, ax=ax3)

    plt.savefig(filename)
    print("Saved results as {}.".format(filename))


if __name__ == '__main__':
    main()
