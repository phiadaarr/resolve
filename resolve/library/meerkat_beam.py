################################################################################
# Copyright (c) 2020, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import io

import numpy as np

KNOWN_MODELS = {
    'MKAT-AA-L-JIM-2020':
    u'''freq, Hx squint, Hy squint, Vx squint, Vy squint, Hx fwhm, Hy fwhm, Vx fwhm, Vy fwhm
         MHz,    arcmin,    arcmin,    arcmin,    arcmin,  arcmin,  arcmin,  arcmin,  arcmin
         900,      0.00,      0.88,     -0.00,      0.72,   97.98,  100.37,   96.41,  101.89
         950,     -0.01,      0.50,     -0.03,      0.41,   92.58,   94.70,   90.72,   96.26
        1000,      0.05,      0.20,      0.02,      0.38,   87.89,   89.30,   85.59,   91.74
        1050,     -0.02,      0.31,      0.02,     -0.12,   83.39,   84.55,   80.96,   86.98
        1100,     -0.03,     -0.03,      0.01,     -0.23,   79.08,   80.06,   76.76,   82.25
        1150,     -0.03,      0.09,     -0.02,     -0.29,   74.79,   76.14,   72.92,   78.09
        1200,     -0.05,      0.00,     -0.00,     -0.36,   70.81,   72.78,   69.70,   73.86
        1250,     -0.05,     -0.03,      0.02,     -0.35,   67.30,   69.69,   66.79,   70.31
        1300,      0.06,      0.02,      0.01,     -0.58,   64.20,   67.18,   64.30,   67.11
        1350,      0.18,     -0.07,      0.03,     -0.42,   61.67,   64.80,   62.15,   64.32
        1400,     -0.43,     -0.07,      0.03,     -0.07,   59.58,   62.70,   60.11,   62.26
        1450,     -1.27,     -0.12,     -0.00,      1.07,   57.92,   60.78,   58.31,   60.45
        1500,     -0.97,     -0.23,     -0.02,      1.14,   56.91,   58.82,   56.61,   59.31
        1550,     -0.40,     -0.21,     -0.02,      0.74,   56.08,   57.00,   54.83,   58.31
        1600,     -0.04,     -0.29,     -0.04,      0.49,   55.35,   55.24,   53.18,   57.44
        1650,      0.22,     -0.18,     -0.04,      1.07,   55.22,   53.52,   51.58,   56.90''',

    'MKAT-AA-UHF-JIM-2020':
    u'''freq, Hx squint, Hy squint, Vx squint, Vy squint, Hx fwhm, Hy fwhm, Vx fwhm, Vy fwhm
         MHz,    arcmin,    arcmin,    arcmin,    arcmin,  arcmin,  arcmin,  arcmin,  arcmin
         550,     -0.15,      2.46,     -0.08,      0.40,  159.05,  165.92,  157.72,  165.00
         600,     -0.00,      0.93,     -0.06,      1.22,  147.75,  153.60,  146.55,  155.25
         650,     -0.02,      1.18,      0.00,      0.43,  135.71,  139.61,  133.70,  141.99
         700,      0.06,      0.14,     -0.01,      0.03,  124.92,  128.66,  122.75,  130.42
         750,      0.07,     -0.13,     -0.02,     -0.16,  115.48,  118.02,  113.03,  121.01
         800,      0.08,     -0.01,      0.02,     -0.81,  106.78,  110.47,  105.56,  111.64
         850,     -0.15,     -0.61,      0.01,     -0.58,   99.25,  103.60,   99.38,  103.52
         900,     -1.12,     -0.61,     -0.01,     -0.10,   93.46,   97.88,   93.96,   97.68
         950,     -1.50,     -0.80,     -0.09,      0.15,   89.67,   93.10,   89.45,   93.52
        1000,     -0.58,     -0.83,     -0.14,     -0.47,   87.38,   88.87,   85.55,   90.83
        1050,      0.32,     -0.43,     -0.08,     -0.72,   86.10,   85.16,   82.32,   88.15'''
}


def _cosine_taper(r):
    # r is normalised such that the half power point occurs at r=0.5:
    # _cosine_taper(0) = 1.0
    # _cosine_taper(0.5) = sqrt(0.5)
    rr = r * 1.18896478
    return np.cos(np.pi * rr) / (1. - 4. * rr**2)


def _pattern(x, y, squint_x, squint_y, fwhm_x, fwhm_y):
    return _cosine_taper(np.sqrt(((x - squint_x) / fwhm_x)**2 + ((y - squint_y) / fwhm_y)**2))

# --------------------------------------------------------------------------------------------------
# --- CLASS :  JimBeam
# --------------------------------------------------------------------------------------------------


class JimBeam:
    """MeerKAT simplified primary beam models for L and UHF bands.

    A cosine aperture taper (Essential Radio Astronomy, Condon & Ransom, 2016,
    page 83, link_) is used as a simplified model of the co-polarisation primary beams.
    While the sidelobe level accuracy may be coincidental, the model attains a good fit
    to measurements for the mainlobe region. The model is parameterised by measured
    frequency dependent pointing, and frequency dependent full width half maximum
    beam widths (FWHM). The MeerKAT beams are measured using holography techniques,
    and an averaged result at 60 degrees elevation is used here to determine the
    frequency dependent parameter values. The pointing errors are determined in
    the aperture plane using standard phase fitting techniques, while the FWHM
    values are measured in the beam plane along axis-aligned cuts through the beam
    centers.

    Parameters
    ----------
    name : str
        Name of model, must be either 'MKAT-AA-L-JIM-2020' or 'MKAT-AA-UHF-JIM-2020'

    Raises
    ------
    ValueError
        If `name` is an unknown model

    Note
    ----
    a) This model is a simplification.
    b) The actual beam varies per antenna, and depends on environmental factors.
    c) Since per-antenna pointing errors during an observation often exceed 1 arc
       minute, the nett 'imaging primary beam' will be slightly broader, and could
       be approximated by averaging several individual antenna beams with
       respective antenna pointing errors inserted.
    d) Depending on the usecase it may be necessary to do reference pointing (or
       use another technique) to remove the antenna pointing errors during the
       observation in order to use a beam model successfully.

    Note
    ----
    As a user, please email the author (mattieu@ska.ac.za) with details about
    your usecase requirements. This may influence future releases. A general
    description, what extent of the beams are needed, pixelation, frequency
    resolution, and accuracy requirements are of interest.

    Example
    -------

    .. code-block:: python

      import matplotlib.pylab as plt
      from resolve import JimBeam
      import numpy as np

      beam = JimBeam('MKAT-AA-UHF-JIM-2020')
      freqMHz = 800
      beamextent = 10
      pol = "H"
      margin = np.linspace(-beamextent/2., beamextent/2., 128)
      x,y = np.meshgrid(margin,margin)
      if pol == "H":
          beampixels = beam.HH(x,y,freqMHz)
      elif pol == "V":
          beampixels = beam.VV(x,y,freqMHz)
      else:
          beampixels = beam.I(x,y,freqMHz)
          pol = "I"
      plt.imshow(beampixels,extent=2*[-beamextent/2, beamextent/2])
      plt.title(f"{pol} pol beam for {beam.name} at {freqMHz}MHz")
      plt.xlabel("deg")
      plt.ylabel("deg")
      plt.show()

    .. _link: https://books.google.co.za/books?id=Jg6hCwAAQBAJ
    """

    def __init__(self, name='MKAT-AA-L-JIM-2020'):
        self.name = name
        try:
            csv_file = io.StringIO(KNOWN_MODELS[name])
        except KeyError:
            raise ValueError('Unknown model {!r}, available ones are {!r}'
                             .format(name, list(KNOWN_MODELS.keys())))
        else:
            table = np.loadtxt(csv_file, skiprows=2, delimiter=',')
        self.freqMHzlist = table[:, 0]
        # Shape (4, nfreq), where 4 refers to Hx,Hy,Vx,Vy components (and arcmin to degrees)
        self.squintlist = table[:, 1:5].T / 60.
        self.fwhmlist = table[:, 5:9].T / 60.

    def _interp_squint_fwhm(self, freqMHz):
        squint = [np.interp(freqMHz, self.freqMHzlist, lst) for lst in self.squintlist]
        fwhm = [np.interp(freqMHz, self.freqMHzlist, lst) for lst in self.fwhmlist]
        return squint, fwhm

    def HH(self, x, y, freqMHz):
        """Calculate the H co-polarised beam at the provided coordinates.
        Parameters
        ----------
        x, y : arrays of float of the same shape
            Coordinates where beam is sampled, in degrees
        freqMHz : float
            Frequency, in MHz
        Returns
        -------
        HH : array of float, same shape as `x` and `y`
            The H co-polarised beam
        """
        squint, fwhm = self._interp_squint_fwhm(freqMHz)
        return _pattern(x, y, squint[0], squint[1], fwhm[0], fwhm[1])

    def VV(self, x, y, freqMHz):
        """Calculate the V co-polarised beam at the provided coordinates.
        Parameters
        ----------
        x, y : arrays of float of the same shape
            Coordinates where beam is sampled, in degrees
        freqMHz : float
            Frequency, in MHz
        Returns
        -------
        VV : array of float, same shape as `x` and `y`
            The V co-polarised beam
        """
        squint, fwhm = self._interp_squint_fwhm(freqMHz)
        return _pattern(x, y, squint[2], squint[3], fwhm[2], fwhm[3])

    def I(self, x, y, freqMHz):  # noqa: E741, E743
        """Calculate the Stokes I beam at the provided coordinates.
        Parameters
        ----------
        x, y : arrays of float of the same shape
            Coordinates where beam is sampled, in degrees
        freqMHz : float
            Frequency, in MHz
        Returns
        -------
        I : array of float, same shape as `x` and `y`
            The Stokes I beam (non-negative)
        """
        H = self.HH(x, y, freqMHz)
        V = self.VV(x, y, freqMHz)
        return 0.5 * (np.abs(H)**2 + np.abs(V)**2)
