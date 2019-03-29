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

import matplotlib.pyplot as plt
import numpy as np

from .likelihood import make_calibration


def data_plot(name, dhs, a1, a2, pol, cal_ops=None, position=None):
    lim = max([np.max(np.abs(dhs['c'].uv)), np.max(np.abs(dhs['t'].uv))])
    fig, axs = plt.subplots(2, 2, figsize=[30, 20])
    for key, dh in dhs.items():
        flag = np.logical_and(dh.ant1 == a1, dh.ant2 == a2)
        vis = dh.vis.val[flag][:, 0, pol]
        var = dh.var.val[flag][:, 0, pol]
        u = dh.uv[flag][:, 0, 0]
        v = dh.uv[flag][:, 0, 1]
        t = dh.time[flag]
        std_abs = np.sqrt(var)
        if cal_ops is not None and position is not None:
            cal = make_calibration(dh, cal_ops)
            vis = (dh.vis/cal.force(position)).val[flag][:, 0, pol]

        axs[0, 1].scatter(t, np.angle(vis, deg=True), label=key)
        axs[0, 1].set_ylim(-180, 180)
        axs[0, 1].set_title('Angle [deg]')

        axs[1, 0].set_title('Absolute part')
        axs[1, 0].errorbar(t, np.abs(vis), yerr=std_abs, fmt='o', label=key)

        axs[1, 1].errorbar(
            t, vis.real, yerr=std_abs, fmt='o', label='Real {}'.format(key))
        axs[1, 1].errorbar(
            t, vis.imag, yerr=std_abs, fmt='o', label='Imag {}'.format(key))
        axs[1, 1].set_title('Real and imag')

        axs[0, 0].set_title('UV plane, antennas {} & {}, pol {}'.format(
            a1, a2, pol))
        axs[0, 0].scatter(u, v, label=key)
        axs[0, 0].set_xlim(-lim, lim)
        axs[0, 0].set_ylim(-lim, lim)
        axs[0, 0].set_aspect(1)
    axs[0, 0].scatter(0, 0, label='(0,0)')
    axs[0, 0].legend()
    axs[0, 1].legend()
    axs[1, 0].legend()
    axs[1, 1].legend()
    plt.tight_layout()
    plt.savefig('{}.png'.format(name))
    plt.close()
