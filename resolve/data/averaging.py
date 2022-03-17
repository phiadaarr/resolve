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
# Copyright(C) 2019-2021 Max-Planck-Society
# Author: Martin Reinecke

import numpy as np


def fair_share_averaging(ts_per_bin, times, gap_time):
    """Bin time stamps fairly.

    Parameters
    ----------
    ts_per_bin : int
        Minimum number of data points in one average.
    times : np.array
        One-dimensional array of time stamps.
    gap_time : float
        Maximum time difference that can occur in one average.
    """
    if ts_per_bin == 1:
        return times
    if ts_per_bin < 1:
        raise ValueError

    times = np.sort(times.copy())
    tbins = []
    nval = len(times)
    i0 = 0

    # The following algorithm was designed to write the same time stamp for a
    # given bin into the original array. Now we need only the bounds of the time
    # bins. Thus, the following implementation could be improved to directly
    # compute these.
    while i0 < nval:
        i = i0+1
        nscan = 1  # number of different time stamps in the scan
        while i < nval and times[i]-times[i-1] < gap_time:  # as long as there are less than x seconds between time stamps, we assume we are in the same scan
            if times[i] != times[i-1]:
                nscan += 1
            i += 1
        nbin = max(1, nscan//ts_per_bin)  # how many bins to use for this scan
        for j in range(nbin):
            n = _fair_share(nscan, nbin, j)
            i = i0+1
            icnt = 0
            oldtime = times[i0]
            while i < nval and icnt < n:
                if times[i] != oldtime:
                    icnt += 1
                    oldtime = times[i]
                if icnt < n:
                    if icnt == n-1:
                        tbins += [(times[i0], times[i], times[i]-times[i0])]
                    times[i] = times[i0]  # give all values in this bin the time stamp of the first value
                    i += 1
            i0 = i
    tbsize = np.array([t[2] for t in tbins])
    print("Size time bins:")
    print(f"  min: {np.min(tbsize):.1f}s\n  max: {np.max(tbsize):.1f}s")
    print(f"  mean: {np.mean(tbsize):.1f}s\n  median: {np.median(tbsize):.1f}s")
    times = np.unique(times)
    times = np.hstack([times, np.inf])
    time_bins = np.array([times[:-1], times[1:]]).T
    return time_bins


def _fair_share(n, nshare, ishare):
    return n//nshare + (ishare < (n % nshare))
