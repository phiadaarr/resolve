# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2020 Max-Planck-Society
# Author: Philipp Arras

import pickle

import nifty7 as ift

from .util import (compare_attributes, my_assert, my_assert_isinstance,
                   my_asserteq)


def simple_minimize(operator, position, n_samples, minimizer, constants=[], point_estimates=[]):
    mini = Minimization(operator, position, n_samples, constants, point_estimates)
    return mini.minimize(minimizer)


class Minimization:
    def __init__(self, operator, position, n_samples, constants=[], point_estimates=[]):
        n_samples = int(n_samples)
        self._position = position
        position = position.extract(operator.domain)
        if n_samples == 0:
            self._e = ift.EnergyAdapter(position, operator, constants, True, True)
            self._n = n_samples
        else:
            my_assert(n_samples > 0)
            dct = {'mean': position,
                   'hamiltonian': operator,
                   'n_samples': n_samples,
                   'constants': constants,
                   'point_estimates': point_estimates,
                   'mirror_samples': True,
                   'comm': None,  # TODO Add MPI support
                   'nanisinf': True
            }
            self._e = ift.MetricGaussianKL.make(**dct)
            self._n, self._m = dct['n_samples'], dct['mirror_samples']

    def minimize(self, minimizer):
        self._e, _ = minimizer(self._e)
        position = ift.MultiField.union([self._position, self._e.position])
        if self._n == 0:
            return MinimizationState(position, [])
        samples = list(self._e.samples)
        my_asserteq(len(samples), 2*self._n if self._m else self._n)
        return MinimizationState(position, samples, self._m)


class MinimizationState:
    def __init__(self, position, samples, mirror_samples=False):
        self._samples = list(samples)
        self._position = position
        if len(samples) > 0:
            my_asserteq(samples[0].domain, *[ss.domain for ss in samples])
            my_assert(set(samples[0].domain.keys()) <= set(position.domain.keys()))
        self._mirror = bool(mirror_samples)

    def __getitem__(self, key):
        # TODO Add MPI support
        if not isinstance(key, int):
            raise TypeError
        if key >= len(self) or key < 0:
            raise KeyError
        if self._mirror and key >= len(self)//2:
            return self._position.unite(-self._samples[key])
        return self._position.unite(self._samples[key])

    def __len__(self):
        return (2 if self._mirror else 1)*len(self._samples)

    def __eq__(self, other):
        if not isinstance(other, MinimizationState):
            return False
        return compare_attributes(self, other, ('_samples', '_position', '_mirror'))

    @property
    def mean(self):
        return self._position

    @property
    def domain(self):
        return self._position.domain

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump([self._position, self._samples, self._mirror],
                        f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(file_name):
        with open(file_name, 'rb') as f:
            position, samples, mirror = pickle.load(f)
        my_assert_isinstance(position, (ift.MultiField, ift.Field))
        my_assert_isinstance(mirror, bool)
        return MinimizationState(position, samples, mirror)
