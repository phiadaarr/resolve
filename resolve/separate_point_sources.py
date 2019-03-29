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
# Copyright(C) 2019 Max-Planck-Society

from starblade import build_starblade, starblade_iteration

import nifty5 as ift
from .extended_operator import ExtendedOperator


def separate_point_sources(diffuse, points, position):
    if not isinstance(diffuse, ExtendedOperator):
        raise TypeError
    if not isinstance(points, ExtendedOperator):
        raise TypeError
    if not isinstance(position, ift.MultiField):
        raise TypeError

    sky = diffuse + points
    tgt = sky.target
    sky_val = sky.force(position).to_global_data()

    # Run starblade
    starblade = build_starblade(data=sky_val, alpha=1.5, cg_steps=5, q=1e-3)
    for i in range(2):
        starblade = starblade_iteration(
            starblade,
            samples=5,
            cg_steps=10,
            newton_steps=100,
            sampling_steps=1000)
    sep_p = starblade.point_like.to_global_data()
    sep_d = starblade.diffuse.to_global_data()

    # Compute pre-images
    diff = diffuse.pre_image(ift.from_global_data(tgt, sep_d))
    pointlike = points.pre_image(ift.from_global_data(tgt, sep_p))
    return ift.MultiField.union([pointlike, diff])
