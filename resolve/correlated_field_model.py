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
# Copyright(C) 2013-2021 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras


class CorrelatedFieldMaker:
    def __init__(self, prefix, total_N=0):
        raise NotImplementedError
    def add_fluctuations(self,
                         target_subdomain,
                         fluctuations,
                         flexibility,
                         asperity,
                         loglogavgslope,
                         prefix='',
                         index=None,
                         dofdex=None,
                         harmonic_partner=None):
        raise NotImplementedError

    def set_amplitude_total_offset(self, offset_mean, offset_std, dofdex=None):
        raise NotImplementedError
    def finalize(self, prior_info=100):
        raise NotImplementedError
