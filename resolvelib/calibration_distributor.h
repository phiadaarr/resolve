/*
 *  This file is part of resolve.
 *
 *  resolve is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  resolve is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with resolve; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2021-2022 Max-Planck-Society, Philipp Arras
   Authors: Philipp Arras */

class CalibrationDistributor {
private:
  const py::str key_logamplitude;
  const py::str key_phase;

  const py::array antenna_indices0;
  const py::array antenna_indices1;
  const py::array time;

  const size_t nfreqs;
  const size_t ntime;
  const double dt;

  size_t nthreads;

  size_t nrows() const { return copy_shape(antenna_indices0)[0]; }

public:
  CalibrationDistributor(const py::array &antenna_indices0_, const py::array &antenna_indices1_,
                         const py::array &time_, const py::str &key_logamplitude_,
                         const py::str &key_phase_, size_t nfreqs_, size_t ntime_, double dt_,
                         size_t nthreads_)
      : key_logamplitude(key_logamplitude_), key_phase(key_phase_),
        antenna_indices0(antenna_indices0_), antenna_indices1(antenna_indices1_), time(time_),
        nfreqs(nfreqs_), ntime(ntime_), dt(dt_), nthreads(nthreads_) {}

  py::array apply(const py::dict &inp_) const {
    // Parse input
    auto logampl = ducc0::to_cmav<double, 4>(inp_[key_logamplitude]);
    auto ph = ducc0::to_cmav<double, 4>(inp_[key_phase]);
    // /Parse input

    // Instantiate output array
    auto npol{ph.shape()[0]};
    auto out_ = ducc0::make_Pyarr<complex<double>>({npol, nrows(), nfreqs});
    auto out = ducc0::to_vmav<complex<double>, 3>(out_);
    // /Instantiate output array

    const auto a0 = ducc0::to_cmav<int, 1>(antenna_indices0);
    const auto a1 = ducc0::to_cmav<int, 1>(antenna_indices1);
    const auto t = ducc0::to_cmav<double, 1>(time);
    for (size_t i0 = 0; i0 < out.shape()[0]; ++i0)
      ducc0::execParallel(out.shape()[1], nthreads, [&](size_t lo, size_t hi) {
        for (size_t i1 = lo; i1 < hi; ++i1)
          for (size_t i2 = 0; i2 < out.shape()[2]; ++i2) {
            const double frac{t(i1) / dt};
            const auto tind0 = size_t(floor(frac));
            const size_t tind1{tind0 + 1};
            MR_assert(tind0 < ntime, "time outside region");
            MR_assert(tind1 < ntime, "time outside region");

            const auto getloggain = [&](const size_t tindex) {
              const auto loggain =
                  complex<double>(logampl(i0, a0(i1), tindex, i2) + logampl(i0, a1(i1), tindex, i2),
                                  ph(i0, a0(i1), tindex, i2) - ph(i0, a1(i1), tindex, i2));
              return loggain;
            };
            const auto diff{frac - double(tind0)};
            const auto loggain{(1 - diff) * getloggain(tind0) + diff * getloggain(tind1)};
            const auto gain{exp(loggain)};
            out(i0, i1, i2) = gain;
          }
      });

    return out_;
  }

  Linearization<py::dict, py::array> apply_with_jac(const py::dict &loc_) {
    // Parse input
    const auto loc_logampl = ducc0::to_cmav<double, 4>(loc_[key_logamplitude]);
    const auto loc_ph = ducc0::to_cmav<double, 4>(loc_[key_phase]);
    const auto inp_shape{loc_ph.shape()};
    // /Parse input

    // Instantiate output array
    auto applied_ = apply(loc_);
    auto applied = ducc0::to_cmav<complex<double>, 3>(applied_);
    // /Instantiate output array

    auto a0 = ducc0::to_cmav<int, 1>(antenna_indices0);
    auto a1 = ducc0::to_cmav<int, 1>(antenna_indices1);
    auto t = ducc0::to_cmav<double, 1>(time);

    function<py::array(const py::dict &)> ftimes = [=](const py::dict &inp_) {
      // Parse input
      const auto inp_logampl = ducc0::to_cmav<double, 4>(inp_[key_logamplitude]);
      const auto inp_ph = ducc0::to_cmav<double, 4>(inp_[key_phase]);
      // /Parse input

      // Instantiate output array
      auto npol{inp_ph.shape()[0]};
      auto out_ = ducc0::make_Pyarr<complex<double>>({npol, nrows(), nfreqs});
      auto out = ducc0::to_vmav<complex<double>, 3>(out_);
      fill_mav(out, complex<double>{0., 0.}, nthreads);
      // /Instantiate output array

      for (size_t i0 = 0; i0 < out.shape()[0]; ++i0)
        ducc0::execParallel(out.shape()[1], nthreads, [&](size_t lo, size_t hi) {
          for (size_t i1 = lo; i1 < hi; ++i1)
            for (size_t i2 = 0; i2 < out.shape()[2]; ++i2) {

              const double frac{t(i1) / dt};
              const auto tind0 = size_t(floor(frac));
              const size_t tind1{tind0 + 1};
              MR_assert(t(i1) >= 0, "time outside region");
              MR_assert(tind0 < ntime, "time outside region");
              MR_assert(tind1 < ntime, "time outside region");

              auto gettmp = [&](const size_t tindex) {
                return applied(i0, i1, i2) *
                       (inp_logampl(i0, a0(i1), tindex, i2) + inp_logampl(i0, a1(i1), tindex, i2) +
                        complex<double>{0, 1} *
                            (inp_ph(i0, a0(i1), tindex, i2) - inp_ph(i0, a1(i1), tindex, i2)));
              };

              const complex<double> tmp0{gettmp(tind0)};
              const complex<double> tmp1{gettmp(tind1)};

              const auto diff{frac - double(tind0)};
              const auto tmp{(1 - diff) * tmp0 + diff * tmp1};

              out(i0, i1, i2) = tmp;
            }
        });
      return out_;
    };

    function<py::dict(const py::array &)> fadjtimes = [=](const py::array &inp_) {
      // Parse input
      auto inp{ducc0::to_cmav<complex<double>, 3>(inp_)};
      // /Parse input

      // Instantiate output
      py::dict out_;
      out_[key_logamplitude] = ducc0::make_Pyarr<double>(inp_shape);
      out_[key_phase] = ducc0::make_Pyarr<double>(inp_shape);
      auto logampl{ducc0::to_vmav<double, 4>(out_[key_logamplitude])};
      auto logph{ducc0::to_vmav<double, 4>(out_[key_phase])};
      fill_mav(logampl, 0., nthreads);
      fill_mav(logph, 0., nthreads);

      for (size_t i0 = 0; i0 < inp.shape()[0]; ++i0)
        for (size_t i1 = 0; i1 < inp.shape()[1]; ++i1)
          for (size_t i2 = 0; i2 < inp.shape()[2]; ++i2) {
            const double frac{t(i1) / dt};
            const auto tind0 = size_t(floor(frac));
            const size_t tind1{tind0 + 1};
            MR_assert(tind0 < ntime, "time outside region");
            MR_assert(tind1 < ntime, "time outside region");

            const auto diff{frac - double(tind0)};

            const auto tmp{conj(applied(i0, i1, i2)) * inp(i0, i1, i2)};
            const auto tmp0{(1 - diff) * tmp};
            const auto tmp1{diff * tmp};

            logampl(i0, a0(i1), tind0, i2) += real(tmp0);
            logampl(i0, a1(i1), tind0, i2) += real(tmp0);
            logph(i0, a0(i1), tind0, i2) += imag(tmp0);
            logph(i0, a1(i1), tind0, i2) -= imag(tmp0);
            logampl(i0, a0(i1), tind1, i2) += real(tmp1);
            logampl(i0, a1(i1), tind1, i2) += real(tmp1);
            logph(i0, a0(i1), tind1, i2) += imag(tmp1);
            logph(i0, a1(i1), tind1, i2) -= imag(tmp1);
          }
      return out_;
    };

    return Linearization<py::dict, py::array>(applied_, ftimes, fadjtimes);
  }
};
