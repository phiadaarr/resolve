# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import nifty7 as ift

from .util import my_assert, my_asserteq


def simple_minimize(operator, position, n_samples, minimizer, constants=[], point_estimates=[]):
    mini = Minimization(operator, position, n_samples, constants, point_estimates)
    return mini.minimize(minimizer)


class Minimization:
    def __init__(self, operator, position, n_samples, constants=[], point_estimates=[]):
        n_samples = int(n_samples)
        position = position.extract(operator.domain)
        if n_samples == 0:
            self._e = ift.EnergyAdapter.make(position, operator, constants, True)
            self._n_samples = n_samples
        else:
            my_assert(n_samples > 0)
            dct = {'position': position,
                   'hamiltonian': operator,
                   'n_samples': n_samples,
                   'constants': constants,
                   'point_estimates': point_estimates,
                   'mirror_samples': self._mirror_samples,
                   'comm': None  # TODO Add MPI support
            }
            self._e = ift.MetricGaussianKL.make(**dct)
            self._n, self._m = dct['n_samples'], dct['mirror_samples']

    def minimize(self, minimizer):
        self._e, _ = minimizer(self._e)
        if self._n_samples == 0:
            samples = SampleStorage([])
        else:
            samples = SampleStorage(self._e.samples, self._m)
            my_asserteq(len(samples), 2*self._n if self._m else self._n)
        return self._e.position, samples


class SampleStorage:
    def __init__(self, samples, mirror_samples=False):
        self._samples = list(samples)
        self._mirror = bool(mirror_samples)

    def __getitem__(self, key):
        # TODO Add MPI support
        if not isinstance(key, int):
            raise TypeError
        if key >= len(self) or key < 0:
            raise KeyError
        if self._mirror and key >= len(self)//2:
            fac = -1
        else:
            fac = 1
        return fac*self._samples[key]

    def __len__(self):
        return (2 if self._mirror else 1)*len(self._samples)
