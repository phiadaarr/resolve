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

#define COMPILE_CFM
#define COMPILE_GAUSSIAN_LIKELIHOOD
#define COMPILE_VARCOV_GAUSSIAN_LIKELIHOOD
#define COMPILE_POLARIZATION_MATRIX_EXPONENTIAL
#define COMPILE_CALIBRATION_DISTRIBUTOR

// FIXME Release GIL around mav_applys
// Includes related to pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/infra/timers.h"
using namespace pybind11::literals;
namespace py = pybind11;
auto None = py::none();
// Includes related to pybind11

#include "ducc0/fft/fft.h"
#include "shape_helper.h"
using namespace std;

// Global types
using Tacc = double;
using Tenergy = double;
using shape_t = vector<size_t>;
// /Global types

template <typename T>
void fill_mav(ducc0::vfmav<T> &mav, const T &val, const size_t nthreads) {
  ducc0::mav_apply([val](T &xx) { xx = val; }, nthreads, mav);
}
template <typename T, size_t ndim>
void fill_mav(ducc0::vmav<T, ndim> &mav, const T &val, const size_t nthreads) {
  ducc0::mav_apply([val](T &xx) { xx = val; }, nthreads, mav);
}

#include "linearization.h"

#ifdef COMPILE_CALIBRATION_DISTRIBUTOR
#include "calibration_distributor.h"
#endif

#ifdef COMPILE_GAUSSIAN_LIKELIHOOD
template <typename T, bool complex_mean,
          typename Tmean = conditional_t<complex_mean, complex<T>, T>>
class DiagonalGaussianLikelihood {
private:
  const size_t nthreads;
  const py::array pymean;
  const py::array pyicov;
  const py::array pymultiplicative;

public:
  const ducc0::cfmav<Tmean> mean;
  const ducc0::cfmav<T> icov;
  const ducc0::cfmav<Tmean> multiplicative;

  DiagonalGaussianLikelihood(const py::array &mean_,
                             const py::array &inverse_covariance_,
                             const py::array &multiplicative_, size_t nthreads_ = 1)
      : nthreads(nthreads_), pymean(mean_), pyicov(inverse_covariance_),
        pymultiplicative(multiplicative_), mean(ducc0::to_cfmav<Tmean>(mean_)),
        icov(ducc0::to_cfmav<T>(inverse_covariance_)),
        multiplicative(ducc0::to_cfmav<Tmean>(pymultiplicative)) {}

private:
  Tenergy value(const py::array &inp) const {
    return value(ducc0::to_cfmav<Tmean>(inp));
  }

  Tenergy value(const ducc0::cfmav<Tmean> &inp) const {
    Tacc acc{0};
    ducc0::mav_apply(
        [&acc](const Tmean &m, const T &ic, const Tmean &l, const Tmean &mult) {
          const auto foo{ic * norm(l * mult - m)};
          const auto foo2 = Tacc(foo);
          acc += foo2;
        },
        1, mean, icov, inp, multiplicative);
    return Tenergy(0.5) * Tenergy(acc);
  }

public:
  py::array apply(const py::array &inp_) const {
    return py::array(py::cast(value(inp_)));
  }

  LinearizationWithMetric<py::array> apply_with_jac(const py::array &loc_) {
    const auto loc{ducc0::to_cfmav<Tmean>(loc_)};
    const auto val = value(loc);

    // gradient
    // Allocate memory with Python to inform Python's GC
    auto gradient_ = py::array_t<Tmean>(loc.shape());
    ducc0::mav_apply(
        [](const Tmean &m, const T &ic, const Tmean &l, const Tmean &mult, Tmean &g) {
          if constexpr (complex_mean) {
            const Tmean tmp2{(l * mult - m) * ic * conj(mult)};
            g = tmp2;
          } else {
            const auto tmp2{(l * mult - m) * ic * mult};
            g = tmp2;
          }
        },
        nthreads, mean, icov, loc, multiplicative, ducc0::to_vfmav<Tmean>(gradient_));
    // /gradient

    // Jacobian
    function<py::array(const py::array &)> ftimes =
        [gradient_ = gradient_](const py::array &inp_) {
          const auto inp{ducc0::to_cfmav<Tmean>(inp_)};
          Tacc acc{0};

          ducc0::mav_apply(
              [&acc](const Tmean &i, const Tmean &g) {
                const T foo{real(i) * real(g) + imag(i) * imag(g)};
                const auto foo2 = Tacc(foo);
                acc += foo2;
              },
              1, inp, ducc0::to_cfmav<Tmean>(gradient_));
          return py::array(py::cast(Tenergy(acc)));
        };
    function<py::array(const py::array &)> fadjtimes =
        [nthreads = nthreads, gradient_ = gradient_](const py::array &inp_) {
          const auto inp{ducc0::to_cfmav<Tenergy>(inp_)};
          const auto inpT{T(inp())};
          const auto gradient = ducc0::to_cfmav<Tmean>(gradient_);
          auto out_{ducc0::make_Pyarr<Tmean>(gradient.shape())};
          auto out{ducc0::to_vfmav<Tmean>(out_)};
          ducc0::mav_apply(
              [inpT = inpT](const Tmean &g, Tmean &o) {
                const Tmean foo{inpT * g};
                o = foo;
              },
              nthreads, gradient, out);
          return out_;
        };
    // /Jacobian

    // Metric
    function<py::array(const py::array &)> apply_metric =
        [nthreads = nthreads, icov = icov,
         multiplicative = multiplicative](const py::array &inp_) {
          const auto inp{ducc0::to_cfmav<Tmean>(inp_)};
          auto out_{ducc0::make_Pyarr<Tmean>(inp.shape())};
          auto out{ducc0::to_vfmav<Tmean>(out_)};
          ducc0::mav_apply(
              [](const Tmean &i, const T &ic, const Tmean &mult, Tmean &o) {
                const Tmean foo{i * ic * norm(mult)};
                o = foo;
              },
              nthreads, inp, icov, multiplicative, out);
          return out_;
        };
    // /Metric

    return LinearizationWithMetric<py::array>(py::array(py::cast(val)), ftimes,
                                              fadjtimes, apply_metric);
  }
};
#endif

#ifdef COMPILE_VARCOV_GAUSSIAN_LIKELIHOOD
template <typename T, bool complex_mean,
          typename Tmean = conditional_t<complex_mean, complex<T>, T>>
class VariableCovarianceDiagonalGaussianLikelihood {
  using Tmask = uint8_t;

private:
  const size_t nthreads;
  const py::array pymean;
  const py::str key_signal;
  const py::str key_log_icov;
  const py::array pymask;

public:
  const ducc0::cfmav<Tmean> mean;
  const ducc0::cfmav<Tmask> mask;

  VariableCovarianceDiagonalGaussianLikelihood(const py::array &mean_,
                                               const py::str &key_signal_,
                                               const py::str &key_log_icov_,
                                               const py::array &mask_,
                                               size_t nthreads_ = 1)
      : nthreads(nthreads_), pymean(mean_), key_signal(key_signal_),
        key_log_icov(key_log_icov_), pymask(mask_), mean(ducc0::to_cfmav<Tmean>(mean_)),
        mask(ducc0::to_cfmav<Tmask>(mask_)) {}

  py::array apply(const py::dict &inp_) const {
    auto signal{ducc0::to_cfmav<Tmean>(inp_[key_signal])};
    auto logicov{ducc0::to_cfmav<T>(inp_[key_log_icov])};
    Tacc acc{0};
    ducc0::mav_apply(
        [&acc](const Tmean &m, const T &lic, const Tmean &l, const Tmask &msk) {
          T fct{1};
          if (complex_mean)
            fct *= 2;
          const Tacc foo{(exp(lic) * norm(l - m) - fct * lic) * T(msk)};
          acc += foo;
        },
        1, mean, logicov, signal,
        mask); // not parallelized because accumulating
    acc *= 0.5;
    return py::array(py::cast(Tenergy(acc)));
  }

  LinearizationWithMetric<py::dict> apply_with_jac(const py::dict &loc_) {
    auto loc_s{ducc0::to_cfmav<Tmean>(loc_[key_signal])};
    auto loc_lic{ducc0::to_cfmav<T>(loc_[key_log_icov])};

    // FIXME Initialize this with python
    auto grad_s{ducc0::vfmav<Tmean>(loc_s.shape(), ducc0::UNINITIALIZED)};
    auto grad_lic{ducc0::vfmav<T>(loc_lic.shape(), ducc0::UNINITIALIZED)};
    Tacc acc{0};

    // value FIXME This is a duplicate
    ducc0::mav_apply(
        [&acc](const Tmean &m, const T &lic, const Tmean &l, const Tmask &msk) {
          T fct{1};
          if (complex_mean)
            fct *= 2;
          const Tacc foo{(exp(lic) * norm(l - m) - fct * lic) * T(msk)};
          acc += foo;
        },
        1, mean, loc_lic, loc_s, mask); // not parallelized because accumulating
    acc *= 0.5;
    auto energy{Tenergy(acc)};
    // /value

    // gradient
    ducc0::mav_apply(
        [](const Tmean &m, const Tmean &s, const T &lic, Tmean &gs, T &glic,
           const Tmask &msk) {
          const auto explic{exp(lic)};
          const auto tmp2{(s - m) * explic * T(msk)};
          T fct;
          if (complex_mean)
            fct = 2;
          else
            fct = 1;
          const T tmp3{T(0.5) * (explic * norm(m - s) - fct) * T(msk)};
          gs = tmp2;
          glic = tmp3;
        },
        nthreads, mean, loc_s, loc_lic, grad_s, grad_lic, mask);
    // /gradient

    // Jacobian
    function<py::array(const py::dict &)> ftimes =
        [grad_s = grad_s, grad_lic = grad_lic, key_signal = key_signal,
         key_log_icov = key_log_icov](const py::dict &inp_) {
          auto inp_s{ducc0::to_cfmav<Tmean>(inp_[key_signal])};
          auto inp_lic{ducc0::to_cfmav<T>(inp_[key_log_icov])};

          Tacc acc{0};

          ducc0::mav_apply(
              [&acc](const Tmean &is, const T &ilic, const Tmean &gs, const T &glic) {
                const Tacc foo{real(is) * real(gs) + imag(is) * imag(gs) + ilic * glic};
                acc += foo;
              },
              1, inp_s, inp_lic, grad_s,
              grad_lic); // not parallelized because accumulating
          return py::array(py::cast(Tenergy(acc)));
        };

    function<py::dict(const py::array &)> fadjtimes =
        [nthreads = nthreads, grad_s = grad_s, grad_lic = grad_lic,
         key_signal = key_signal, key_log_icov = key_log_icov](const py::array &inp_) {
          auto inp{ducc0::to_cfmav<Tenergy>(inp_)};
          auto inpT{T(inp())};
          py::dict out_;
          out_[key_signal] = ducc0::make_Pyarr<Tmean>(grad_s.shape());
          out_[key_log_icov] = ducc0::make_Pyarr<T>(grad_s.shape());
          auto outs{ducc0::to_vfmav<Tmean>(out_[key_signal])};
          auto outlic{ducc0::to_vfmav<T>(out_[key_log_icov])};
          ducc0::mav_apply(
              [inpT = inpT](const Tmean &gs, const T &glic, Tmean &os, T &olic) {
                const Tmean foo{inpT * gs};
                const T foo2{inpT * glic};
                os = foo;
                olic = foo2;
              },
              nthreads, grad_s, grad_lic, outs, outlic);
          return out_;
        };
    // /Jacobian

    // Metric
    function<py::dict(const py::dict &)> apply_metric =
        [nthreads = nthreads, key_signal = key_signal, key_log_icov = key_log_icov,
         loc_lic = loc_lic, mask = mask](const py::dict &inp_) {
          auto inp_s{ducc0::to_cfmav<Tmean>(inp_[key_signal])};
          auto inp_lic{ducc0::to_cfmav<T>(inp_[key_log_icov])};

          py::dict out_;
          out_[key_signal] = ducc0::make_Pyarr<Tmean>(inp_s.shape());
          out_[key_log_icov] = ducc0::make_Pyarr<T>(inp_s.shape());
          auto outs{ducc0::to_vfmav<Tmean>(out_[key_signal])};
          auto outlic{ducc0::to_vfmav<T>(out_[key_log_icov])};

          ducc0::mav_apply(
              [](const T &lic, const Tmean &ins, const T &inlic, Tmean &os, T &olic,
                 const Tmask &msk) {
                T fac{1};
                if (!complex_mean)
                  fac *= T(0.5);
                const Tmean foo{exp(lic) * ins * T(msk)};
                const T foo2{inlic * T(msk) * fac};
                os = foo;
                olic = foo2;
              },
              nthreads, loc_lic, inp_s, inp_lic, outs, outlic, mask);
          return out_;
        };
    // /Metric

    return LinearizationWithMetric<py::dict>(py::array(py::cast(energy)), ftimes,
                                             fadjtimes, apply_metric);
  }
};
#endif

#ifdef COMPILE_POLARIZATION_MATRIX_EXPONENTIAL
template <typename T, size_t ndim> class PolarizationMatrixExponential {
private:
  const size_t nthreads;

public:
  PolarizationMatrixExponential(size_t nthreads_ = 1) : nthreads(nthreads_) {}

  py::array apply(const py::dict &inp_) const {
    // Parse input
    auto I{ducc0::to_cmav<T, ndim>(inp_["I"])}, Q{ducc0::to_cmav<T, ndim>(inp_["Q"])},
        U{ducc0::to_cmav<T, ndim>(inp_["U"])}, V{ducc0::to_cmav<T, ndim>(inp_["V"])};
    // /Parse input

    // Instantiate output array
    auto out_ = ducc0::make_Pyarr<T>(combine_shapes(4, I.shape()));
    auto out = ducc0::to_vmav<T, ndim + 1>(out_);
    vector<ducc0::slice> slcs(ndim + 1);
    slcs[0] = ducc0::slice(0);
    auto outI = ducc0::subarray<ndim>(out, slcs);
    slcs[0] = ducc0::slice(1);
    auto outQ = ducc0::subarray<ndim>(out, slcs);
    slcs[0] = ducc0::slice(2);
    auto outU = ducc0::subarray<ndim>(out, slcs);
    slcs[0] = ducc0::slice(3);
    auto outV = ducc0::subarray<ndim>(out, slcs);
    // /Instantiate output array

    ducc0::mav_apply(
        [](const auto &ii, const auto &qq, const auto &uu, const auto &vv, auto &oii,
           auto &oqq, auto &ouu, auto &ovv) {
          const auto pol{sqrt(qq * qq + uu * uu + vv * vv)};
          const auto expii{exp(ii)};
          const auto exppol{exp(pol)};
          const auto tmp{0.5 * expii / pol * (exppol - 1. / exppol)};
          const auto tmpi{0.5 * expii * (exppol + 1. / exppol)};
          const auto tmpq{tmp * qq};
          const auto tmpu{tmp * uu};
          const auto tmpv{tmp * vv};
          oii = tmpi;
          oqq = tmpq;
          ouu = tmpu;
          ovv = tmpv;
        },
        nthreads, I, Q, U, V, outI, outQ, outU, outV);
    return out_;
  }

  Linearization<py::dict, py::array> apply_with_jac(const py::dict &loc_) {
    // Parse input
    auto I{ducc0::to_cmav<T, ndim>(loc_["I"])}, Q{ducc0::to_cmav<T, ndim>(loc_["Q"])},
        U{ducc0::to_cmav<T, ndim>(loc_["U"])}, V{ducc0::to_cmav<T, ndim>(loc_["V"])};
    // /Parse input

    // Instantiate output array
    auto applied_ = ducc0::make_Pyarr<T>(combine_shapes(4, I.shape()));
    auto applied = ducc0::to_vmav<T, ndim + 1>(applied_);

    vector<ducc0::slice> slcs(ndim + 1);
    slcs[0] = ducc0::slice(0);
    auto appliedI = ducc0::subarray<ndim>(applied, slcs);
    slcs[0] = ducc0::slice(1);
    auto appliedQ = ducc0::subarray<ndim>(applied, slcs);
    slcs[0] = ducc0::slice(2);
    auto appliedU = ducc0::subarray<ndim>(applied, slcs);
    slcs[0] = ducc0::slice(3);
    auto appliedV = ducc0::subarray<ndim>(applied, slcs);
    // /Instantiate output array

    struct mtx {
      T iq, iu, iv, qq, qu, qv, uq, uu, uv, vq, vu, vv;
    };

    // Allocate Jacobian
    // Initialize this with python?
    auto mat = ducc0::vmav<mtx, ndim>(I.shape());
    // /Allocate Jacobian

    // Derive + apply
    ducc0::mav_apply(
        [](const auto &ii, const auto &qq, const auto &uu, const auto &vv, auto &dii,
           auto &dqi, auto &dui, auto &dvi, auto &d) {
          const auto pol0{qq * qq + uu * uu + vv * vv};
          const auto pol{sqrt(pol0)};
          const auto expii{exp(ii)};
          const auto exppol{exp(pol)};
          const auto eplus0{expii * exppol}, eminus0{expii / exppol};
          const auto tmp2{0.5 / pol * (eplus0 - eminus0)};
          dii = 0.5 * (eplus0 + eminus0);
          d.iq = tmp2 * qq;
          d.iu = tmp2 * uu;
          d.iv = tmp2 * vv;

          const auto eplus{(expii / pol) * exppol};
          const auto eminus{(expii / pol) / exppol};
          const auto tmp{0.5 * (eplus - eminus)};
          const auto tmp3{0.5 / pol0 * (eplus * (pol - 1.) + eminus * (pol + 1.))};

          const auto tmpq{tmp3 * qq};
          dqi = tmp * qq;
          d.qq = qq * tmpq + tmp;
          d.qu = uu * tmpq;
          d.qv = vv * tmpq;

          const auto tmpu{tmp3 * uu};
          dui = tmp * uu;
          d.uq = qq * tmpu;
          d.uu = uu * tmpu + tmp;
          d.uv = vv * tmpu;

          const auto tmpv{tmp3 * vv};
          dvi = tmp * vv;
          d.vq = qq * tmpv;
          d.vu = uu * tmpv;
          d.vv = vv * tmpv + tmp;
        },
        nthreads, I, Q, U, V, appliedI, appliedQ, appliedU, appliedV, mat);
    // /Derive + apply

    function<py::array(const py::dict &)> ftimes = [nthreads = nthreads, appliedI,
                                                    appliedQ, appliedU, appliedV,
                                                    mat](const py::dict &inp_) {
      // Parse input
      auto I{ducc0::to_cmav<T, ndim>(inp_["I"])}, Q{ducc0::to_cmav<T, ndim>(inp_["Q"])},
          U{ducc0::to_cmav<T, ndim>(inp_["U"])}, V{ducc0::to_cmav<T, ndim>(inp_["V"])};
      // /Parse input

      // Instantiate output array
      auto out_ = ducc0::make_Pyarr<T>(combine_shapes(4, I.shape()));
      auto out = ducc0::to_vmav<T, ndim + 1>(out_);
      vector<ducc0::slice> slcs(ndim + 1);
      slcs[0] = ducc0::slice(0);
      auto outI = ducc0::subarray<ndim>(out, slcs);
      slcs[0] = ducc0::slice(1);
      auto outQ = ducc0::subarray<ndim>(out, slcs);
      slcs[0] = ducc0::slice(2);
      auto outU = ducc0::subarray<ndim>(out, slcs);
      slcs[0] = ducc0::slice(3);
      auto outV = ducc0::subarray<ndim>(out, slcs);
      // /Instantiate output array

      // Matrix multiplication
      ducc0::mav_apply(
          [](const auto &dii, const auto &dqi, const auto &dui, const auto &dvi,
             const auto &d, const auto &ii, const auto &qq, const auto &uu,
             const auto &vv, auto &iiout, auto &qqout, auto &uuout, auto &vvout) {
            const auto ti = dii * ii + d.iq * qq + d.iu * uu + d.iv * vv;
            const auto tq = dqi * ii + d.qq * qq + d.qu * uu + d.qv * vv;
            const auto tu = dui * ii + d.uq * qq + d.uu * uu + d.uv * vv;
            const auto tv = dvi * ii + d.vq * qq + d.vu * uu + d.vv * vv;
            iiout = ti;
            qqout = tq;
            uuout = tu;
            vvout = tv;
          },
          nthreads, appliedI, appliedQ, appliedU, appliedV, mat, I, Q, U, V, outI, outQ,
          outU, outV);
      // /Matrix multiplication

      return out_;
    };

    function<py::dict(const py::array &)> fadjtimes = [nthreads = nthreads, appliedI,
                                                       appliedQ, appliedU, appliedV,
                                                       mat](const py::array &inp_) {
      // Parse input
      auto inp{ducc0::to_cmav<T, ndim + 1>(inp_)};
      vector<ducc0::slice> slcs(ndim + 1);
      slcs[0] = ducc0::slice(0);
      auto I{ducc0::subarray<ndim>(inp, slcs)};
      slcs[0] = ducc0::slice(1);
      auto Q{ducc0::subarray<ndim>(inp, slcs)};
      slcs[0] = ducc0::slice(2);
      auto U{ducc0::subarray<ndim>(inp, slcs)};
      slcs[0] = ducc0::slice(3);
      auto V{ducc0::subarray<ndim>(inp, slcs)};
      // /Parse input

      // Instantiate output
      py::dict out_;
      out_["I"] = ducc0::make_Pyarr<T>(I.shape());
      out_["Q"] = ducc0::make_Pyarr<T>(I.shape());
      out_["U"] = ducc0::make_Pyarr<T>(I.shape());
      out_["V"] = ducc0::make_Pyarr<T>(I.shape());
      auto outI{ducc0::to_vmav<T, ndim>(out_["I"])},
          outQ{ducc0::to_vmav<T, ndim>(out_["Q"])},
          outU{ducc0::to_vmav<T, ndim>(out_["U"])},
          outV{ducc0::to_vmav<T, ndim>(out_["V"])};
      // /Instantiate output

      // Adjoint matrix multiplication
      ducc0::mav_apply(
          [](const auto &dii, const auto &dqi, const auto &dui, const auto &dvi,
             const auto &d, const auto &ii, const auto &qq, const auto &uu,
             const auto &vv, auto &iiout, auto &qqout, auto &uuout, auto &vvout) {
            const auto ti = dii * ii + dqi * qq + dui * uu + dvi * vv;
            const auto tq = d.iq * ii + d.qq * qq + d.uq * uu + d.vq * vv;
            const auto tu = d.iu * ii + d.qu * qq + d.uu * uu + d.vu * vv;
            const auto tv = d.iv * ii + d.qv * qq + d.uv * uu + d.vv * vv;
            iiout = ti;
            qqout = tq;
            uuout = tu;
            vvout = tv;
          },
          nthreads, appliedI, appliedQ, appliedU, appliedV, mat, I, Q, U, V, outI, outQ,
          outU, outV);
      // /Adjoint matrix multiplication
      return out_;
    };

    return Linearization<py::dict, py::array>(applied_, ftimes, fadjtimes);
  }
};
#endif

#ifdef COMPILE_CFM
class CfmCore {
private:
  const py::list amplitude_keys;
  const py::list pindices;
  const py::str key_xi;
  const py::str key_azm;
  const double offset_mean;
  const double scalar_dvol;
  vector<ducc0::cfmav<int64_t>> lpindex;
  shape_t dimlim;
  shape_t space_dims;
  const size_t n_pspecs;

  size_t nthreads;

  shape_t fft_axes(const size_t &ispace) const {
    shape_t out;
    for (size_t idim = dimlim[ispace]; idim < dimlim[ispace + 1]; ++idim)
      out.emplace_back(idim);
    return out;
  }

public:
  CfmCore(const py::list &pindices_, const py::list &amplitude_keys_,
          const py::str &key_xi_, const py::str &key_azm_, const double &offset_mean_,
          const double &scalar_dvol_, const size_t nthreads_)
      : amplitude_keys(amplitude_keys_), pindices(pindices_), key_xi(key_xi_),
        key_azm(key_azm_), offset_mean(offset_mean_), scalar_dvol(scalar_dvol_),
        n_pspecs(py::len(amplitude_keys)), nthreads(nthreads_) {
    dimlim.push_back(1);
    for (size_t i = 0; i < n_pspecs; ++i) {
      lpindex.push_back(ducc0::to_cfmav<int64_t>(pindices[i]));
      dimlim.push_back(dimlim.back() + lpindex.back().ndim());
      space_dims.push_back(lpindex.back().ndim());
    }
  }

  vector<ducc0::cfmav<double>>
  precompute_distributed_spectra(const py::dict &inp_) const {
    const auto out_shape{copy_shape(inp_[key_xi])};
    const size_t total_N = out_shape[0];
    vector<ducc0::cfmav<double>> distributed_power_spectra;
    for (size_t i = 0; i < n_pspecs; ++i) {
      const auto inp_pspec(ducc0::to_cmav<double, 2>(inp_[amplitude_keys[i]]));
      const auto lshape = combine_shapes(total_N, lpindex[i].shape());
      ducc0::vfmav<double> tmp(lshape);
      const size_t actual_dimensions = space_dims[i];
      ducc0::mav_apply_with_index(
          [&](double &oo, const shape_t &inds) {
            const int64_t pind = lpindex[i].val(&inds[1], &inds[actual_dimensions + 1]);
            oo = inp_pspec(inds[0], pind);
          },
          nthreads, tmp);

      ducc0::fmav_info::shape_t axpos;
      axpos.push_back(0);
      for (size_t j = dimlim[i]; j < dimlim[i + 1]; ++j)
        axpos.push_back(j);
      distributed_power_spectra.emplace_back(
          tmp.extend_and_broadcast(out_shape, axpos));
    }
    return distributed_power_spectra;
  }

  void A_times_xi(ducc0::cfmav<double> inp_xi, ducc0::cfmav<double> inp_azm,
                  const vector<ducc0::cfmav<double>> &distributed_power_spectra,
                  ducc0::vfmav<double> &out) const {
    auto inp_azm_broadcast = inp_azm.extend_and_broadcast(inp_xi.shape(), 0);

    if (n_pspecs == 1)
      ducc0::mav_apply([&](double &oo, const double &azm, const double &xi,
                           const double &s0) { oo = azm * xi * s0; },
                       nthreads, out, inp_azm_broadcast, inp_xi,
                       distributed_power_spectra[0]);
    else if (n_pspecs == 2)
      ducc0::mav_apply([&](double &oo, const double &azm, const double &xi,
                           const double &s0,
                           const double &s1) { oo = azm * xi * s0 * s1; },
                       nthreads, out, inp_azm_broadcast, inp_xi,
                       distributed_power_spectra[0], distributed_power_spectra[1]);
    // and so on, as far as we want to go with special cases
    else
      ducc0::mav_apply_with_index(
          [&](double &oo, const double &xi, const shape_t &inds) {
            double foop{1};
            for (size_t i = 0; i < n_pspecs; ++i) {
              foop *= distributed_power_spectra[i](inds);
            }
            const double foozm{inp_azm(inds[0]) * xi};
            oo = foozm * foop;
          },
          nthreads, out, inp_xi);
  }

  void
  A_times_xi_jac(const ducc0::cfmav<double> inp_xi, const ducc0::cfmav<double> inp_azm,
                 const vector<ducc0::cfmav<double>> &inp_distributed_power_spectra,
                 const ducc0::cfmav<double> tangent_xi,
                 const ducc0::cfmav<double> tangent_azm,
                 const vector<ducc0::cfmav<double>> &tangent_distributed_power_spectra,
                 ducc0::vfmav<double> &out) const {

    auto inp_azm_broadcast = inp_azm.extend_and_broadcast(inp_xi.shape(), 0);
    auto tangent_azm_broadcast = tangent_azm.extend_and_broadcast(inp_xi.shape(), 0);

    if (n_pspecs == 1)
      ducc0::mav_apply(
          [&](double &out, const double &inp_xi, const double &inp_azm_broadcast,
              const double &tangent_xi, const double &tangent_azm_broadcast,
              const double &s0, const double &ts0) {
            const auto term0 = ts0 * inp_azm_broadcast * inp_xi;
            const auto term1 = s0 * tangent_azm_broadcast * inp_xi;
            const auto term2 = s0 * inp_azm_broadcast * tangent_xi;
            out = term0 + term1 + term2;
          },
          nthreads, out, inp_xi, inp_azm_broadcast, tangent_xi, tangent_azm_broadcast,
          inp_distributed_power_spectra[0], tangent_distributed_power_spectra[0]);
    else if (n_pspecs == 2)
      ducc0::mav_apply(
          [&](double &out, const double &inp_xi, const double &inp_azm_broadcast,
              const double &tangent_xi, const double &tangent_azm_broadcast,
              const double &s0, const double &s1, const double &ts0,
              const double &ts1) {
            double inp_pspec{s0 * s1};

            double dpspec = s1 * ts0 + s0 * ts1;

            const auto term0 = dpspec * inp_azm_broadcast * inp_xi;
            const auto term1 = inp_pspec * tangent_azm_broadcast * inp_xi;
            const auto term2 = inp_pspec * inp_azm_broadcast * tangent_xi;
            out = term0 + term1 + term2;
          },
          nthreads, out, inp_xi, inp_azm_broadcast, tangent_xi, tangent_azm_broadcast,
          inp_distributed_power_spectra[0], inp_distributed_power_spectra[1],
          tangent_distributed_power_spectra[0], tangent_distributed_power_spectra[1]);
    // [...]
    else
      ducc0::mav_apply_with_index(
          [&](double &out, const double &inp_xi, const double &inp_azm_broadcast,
              const double &tangent_xi, const double &tangent_azm_broadcast,
              const shape_t &inds) {
            double inp_pspec{1.};
            for (size_t i = 0; i < n_pspecs; ++i) {
              inp_pspec *= inp_distributed_power_spectra[i](inds);
            }

            double dpspec{0};
            for (size_t i = 0; i < n_pspecs; ++i) {
              const auto tmp{inp_pspec / inp_distributed_power_spectra[i](inds) *
                             tangent_distributed_power_spectra[i](inds)};
              dpspec += tmp;
            }

            const auto term0 = dpspec * inp_azm_broadcast * inp_xi;
            const auto term1 = inp_pspec * tangent_azm_broadcast * inp_xi;
            const auto term2 = inp_pspec * inp_azm_broadcast * tangent_xi;
            out = term0 + term1 + term2;
          },
          nthreads, out, inp_xi, inp_azm_broadcast, tangent_xi, tangent_azm_broadcast);
  }

  void A_times_xi_adj_jac(
      const ducc0::cfmav<double> inp_xi, const ducc0::cfmav<double> inp_azm,
      const vector<ducc0::cfmav<double>> &inp_distributed_power_spectra,
      const ducc0::vfmav<double> &cotangent_in, ducc0::vfmav<double> &out_xi,
      ducc0::vfmav<double> &out_azm, vector<ducc0::vfmav<double>> &out_pspecs) const {

    const auto inp_azm_broadcast = inp_azm.extend_and_broadcast(inp_xi.shape(), 0);
    const auto out_azm_broadcast = out_azm.extend_and_broadcast(inp_xi.shape(), 0);

    if (n_pspecs <= 2) {
      const auto out_shape{inp_xi.shape()};
      const size_t total_N = out_shape[0];
      vector<ducc0::vfmav<double>> power_spectra_out;
      vector<ducc0::vfmav<double>> distributed_power_spectra_out;
      for (size_t i = 0; i < n_pspecs; ++i) {
        const auto lshape = combine_shapes(total_N, lpindex[i].shape());
        ducc0::vfmav<double> tmp(lshape);
        fill_mav(tmp, 0., nthreads);
        power_spectra_out.emplace_back(tmp);
        ducc0::fmav_info::shape_t axpos;
        axpos.push_back(0);
        for (size_t j = dimlim[i]; j < dimlim[i + 1]; ++j)
          axpos.push_back(j);
        distributed_power_spectra_out.emplace_back(
            tmp.extend_and_broadcast(out_shape, axpos));
      }

      if (n_pspecs == 1)
        ducc0::mav_apply(
            [](const double &inp_xi, const double &inp_azm_broadcast,
               const double &cotangent_in, double &out_xi, double &out_azm_broadcast,
               const double &s0, double &os0) {
              // Note: it shall be supported that cotangent_in and out_xi points to
              // the same memory. So out_xi is written to at the very end
              double d_os0 = inp_xi * inp_azm_broadcast * cotangent_in;
              double d_out_azm_broadcast = inp_xi * cotangent_in * s0;
              out_xi = inp_azm_broadcast * cotangent_in * s0;
              os0 += d_os0;
              out_azm_broadcast += d_out_azm_broadcast;
            },
            1, inp_xi, inp_azm_broadcast, cotangent_in, out_xi, out_azm_broadcast,
            inp_distributed_power_spectra[0], distributed_power_spectra_out[0]);
      else if (n_pspecs == 2)
        ducc0::mav_apply(
            [](const double &inp_xi, const double &inp_azm_broadcast,
               const double &cotangent_in, double &out_xi, double &out_azm_broadcast,
               const double &s0, const double &s1, double &os0, double &os1) {
              // Note: it shall be supported that cotangent_in and out_xi points to
              // the same memory. So out_xi is written to at the very end
              double d_os0 = inp_xi * inp_azm_broadcast * cotangent_in * s1;
              double d_os1 = inp_xi * inp_azm_broadcast * cotangent_in * s0;
              double d_out_azm_broadcast = inp_xi * cotangent_in * s0 * s1;
              out_xi = inp_azm_broadcast * cotangent_in * s0 * s1;
              os0 += d_os0;
              os1 += d_os1;
              out_azm_broadcast += d_out_azm_broadcast;
            },
            1, inp_xi, inp_azm_broadcast, cotangent_in, out_xi, out_azm_broadcast,
            inp_distributed_power_spectra[0], inp_distributed_power_spectra[1],
            distributed_power_spectra_out[0], distributed_power_spectra_out[1]);

      vector<int64_t> i0(out_xi.shape(0));
      iota(i0.begin(), i0.end(), 0);
      ducc0::cfmav<int64_t> i0mav(i0.data(), {out_xi.shape(0)});
      for (size_t i = 0; i < n_pspecs; ++i) {
        auto i0mav2(i0mav.extend_and_broadcast(power_spectra_out[i].shape(), 0));
        auto xpindex = lpindex[i].extend_and_broadcast(power_spectra_out[i].shape(), 1);
        ducc0::vmav<double, 2> out_ps(out_pspecs[i]);
        ducc0::mav_apply([&](const double &ii, const int64_t &ind0,
                             const int64_t &pind) { out_ps(ind0, pind) += ii; },
                         1, power_spectra_out[i], i0mav2, xpindex);
      }
    } else
      ducc0::mav_apply_with_index(
          [&](const double &inp_xi, const double &inp_azm_broadcast,
              const double &cotangent_in, double &out_xi, double &out_azm_broadcast,
              const shape_t &inds) {
            // Note: it shall be supported that cotangent_in and out_xi points to
            // the same memory. So out_xi is written to at the very end
            double inp{inp_xi * inp_azm_broadcast * cotangent_in};
            for (size_t i = 0; i < n_pspecs; ++i) {
              inp *= inp_distributed_power_spectra[i](inds);
            }
            for (size_t i = 0; i < n_pspecs; ++i) {
              const auto pspec_index =
                  lpindex[i].val(&inds[dimlim[i]], &inds[dimlim[i + 1]]);
              out_pspecs[i](inds[0], pspec_index) +=
                  inp / inp_distributed_power_spectra[i](inds);
            }
            out_azm_broadcast += inp / inp_azm_broadcast;
            out_xi = inp / inp_xi;
          },
          1, inp_xi, inp_azm_broadcast, cotangent_in, out_xi, out_azm_broadcast);
  }

  void add_offset_mean(const double &offset_mean, ducc0::vfmav<double> &out) const {
    vector<ducc0::slice> myslc(out.ndim(), 0);
    myslc[0] = ducc0::slice();
    ducc0::mav_apply([&](auto &oo) { oo += offset_mean; }, 1, out.subarray(myslc));
  }

  void fft(ducc0::vfmav<double> &out) const {
    double fct{scalar_dvol};
    for (size_t ispace = 0; ispace < n_pspecs; ++ispace) {
      ducc0::r2r_genuine_hartley(out, out, fft_axes(ispace), fct, nthreads);
      fct = 1;
    }
  }

  void fft_adjoint(const ducc0::cfmav<double> &in, ducc0::vfmav<double> &out) const {
    for (size_t i = 0; i < n_pspecs; ++i) {
      MR_assert(i + 1 <= n_pspecs, "this is a bug");
      const size_t ispace{n_pspecs - (i + 1)};
      if (i == 0)
        ducc0::r2r_genuine_hartley(in, out, fft_axes(ispace), scalar_dvol, nthreads);
      else
        ducc0::r2r_genuine_hartley(out, out, fft_axes(ispace), double(1), nthreads);
    }
  }

  py::array apply(const py::dict &inp_) const {
    const auto out_shape = copy_shape(inp_[key_xi]);
    MR_assert(py::len(amplitude_keys) == n_pspecs,
              "Number of input pspecs not equal to length of pindex list");
    auto out_ = ducc0::make_Pyarr<double>(out_shape);
    auto out = ducc0::to_vfmav<double>(out_);
    // Note: "out" is not nulled at this point

    const auto inp_xi = ducc0::to_cfmav<double>(inp_[key_xi]);
    const auto inp_azm = ducc0::to_cfmav<double>(inp_[key_azm]);
    const auto inp_distributed_power_spectra = precompute_distributed_spectra(inp_);
    A_times_xi(inp_xi, inp_azm, inp_distributed_power_spectra, out);

    add_offset_mean(offset_mean, out);
    fft(out);
    return out_;
  }

  Linearization<py::dict, py::array> apply_with_jac(const py::dict &inp_) {
    const auto out_shape = copy_shape(inp_[key_xi]);
    MR_assert(py::len(amplitude_keys) == n_pspecs,
              "Number of input pspecs not equal to length of pindex list");

    // FIXME It would be cleaner to instantiate this inside A_times_xi
    auto out_ = ducc0::make_Pyarr<double>(out_shape);
    auto out = ducc0::to_vfmav<double>(out_);
    // Note: "out" is not nulled at this point

    const auto inp_xi = ducc0::to_cfmav<double>(inp_[key_xi]);
    const auto inp_azm = ducc0::to_cfmav<double>(inp_[key_azm]);
    const auto inp_distributed_power_spectra = precompute_distributed_spectra(inp_);

    A_times_xi(inp_xi, inp_azm, inp_distributed_power_spectra, out);
    add_offset_mean(offset_mean, out);
    fft(out);

    function<py::array(const py::dict &)> ftimes = [=](const py::dict &tangent_) {
      // keep inp_ alive to avoid dangling references
      const auto inpcopy = inp_;
      auto out_ = ducc0::make_Pyarr<double>(out_shape);
      auto out = ducc0::to_vfmav<double>(out_);

      const auto tangent_xi = ducc0::to_cfmav<double>(tangent_[key_xi]);
      const auto tangent_azm = ducc0::to_cfmav<double>(tangent_[key_azm]);
      const auto tangent_distributed_power_spectra =
          precompute_distributed_spectra(tangent_);

      A_times_xi_jac(inp_xi, inp_azm, inp_distributed_power_spectra, tangent_xi,
                     tangent_azm, tangent_distributed_power_spectra, out);

      fft(out);
      return out_;
    };

    function<py::dict(const py::array &)> fadjtimes = [=](const py::array &cotangent_) {
      // keep inp_ alive to avoid dangling references
      const auto inpcopy = inp_;

      const auto cotangent = ducc0::to_cfmav<double>(cotangent_);

      // Instantiate output
      py::dict out_;
      out_[key_xi] = ducc0::make_Pyarr<double>(out_shape);
      auto out_xi = ducc0::to_vfmav<double>(out_[key_xi]);
      // fill_mav(out_xi, 0, nthreads);

      out_[key_azm] = ducc0::make_Pyarr<double>(inp_azm.shape());
      auto out_azm = ducc0::to_vfmav<double>(out_[key_azm]);
      fill_mav(out_azm, 0., nthreads);

      vector<ducc0::vfmav<double>> out_pspec;
      for (size_t i = 0; i < n_pspecs; ++i) {
        out_[amplitude_keys[i]] =
            ducc0::make_Pyarr<double>(copy_shape(inp_[amplitude_keys[i]]));
        out_pspec.push_back(ducc0::to_vfmav<double>(out_[amplitude_keys[i]]));
        fill_mav(out_pspec.back(), 0., nthreads);
      }
      // /Instantiate output

      fft_adjoint(cotangent, out_xi);

      A_times_xi_adj_jac(inp_xi, inp_azm, inp_distributed_power_spectra, out_xi, out_xi,
                         out_azm, out_pspec);

      return out_;
    };

    return Linearization<py::dict, py::array>(out_, ftimes, fadjtimes);
  }
};
#endif

PYBIND11_MODULE(resolve_support, m) {

  m.attr("__name__") = "resolve_support";

#ifdef COMPILE_POLARIZATION_MATRIX_EXPONENTIAL
  py::class_<PolarizationMatrixExponential<double, 1>>(m,
                                                       "PolarizationMatrixExponential1")
      .def(py::init<size_t>())
      .def("apply", &PolarizationMatrixExponential<double, 1>::apply)
      .def("apply_with_jac", &PolarizationMatrixExponential<double, 1>::apply_with_jac);
  py::class_<PolarizationMatrixExponential<double, 2>>(m,
                                                       "PolarizationMatrixExponential2")
      .def(py::init<size_t>())
      .def("apply", &PolarizationMatrixExponential<double, 2>::apply)
      .def("apply_with_jac", &PolarizationMatrixExponential<double, 2>::apply_with_jac);
  py::class_<PolarizationMatrixExponential<double, 3>>(m,
                                                       "PolarizationMatrixExponential3")
      .def(py::init<size_t>())
      .def("apply", &PolarizationMatrixExponential<double, 3>::apply)
      .def("apply_with_jac", &PolarizationMatrixExponential<double, 3>::apply_with_jac);
  py::class_<PolarizationMatrixExponential<double, 4>>(m,
                                                       "PolarizationMatrixExponential4")
      .def(py::init<size_t>())
      .def("apply", &PolarizationMatrixExponential<double, 4>::apply)
      .def("apply_with_jac", &PolarizationMatrixExponential<double, 4>::apply_with_jac);
#endif

#ifdef COMPILE_GAUSSIAN_LIKELIHOOD
  py::class_<DiagonalGaussianLikelihood<double, false>>(m,
                                                        "DiagonalGaussianLikelihood_f8")
      .def(py::init<py::array, py::array, py::array, size_t>())
      .def("apply", &DiagonalGaussianLikelihood<double, false>::apply)
      .def("apply_with_jac",
           &DiagonalGaussianLikelihood<double, false>::apply_with_jac);
  py::class_<DiagonalGaussianLikelihood<float, false>>(m,
                                                       "DiagonalGaussianLikelihood_f4")
      .def(py::init<py::array, py::array, py::array, size_t>())
      .def("apply", &DiagonalGaussianLikelihood<float, false>::apply)
      .def("apply_with_jac", &DiagonalGaussianLikelihood<float, false>::apply_with_jac);
  py::class_<DiagonalGaussianLikelihood<double, true>>(m,
                                                       "DiagonalGaussianLikelihood_c16")
      .def(py::init<py::array, py::array, py::array, size_t>())
      .def("apply", &DiagonalGaussianLikelihood<double, true>::apply)
      .def("apply_with_jac", &DiagonalGaussianLikelihood<double, true>::apply_with_jac);
  py::class_<DiagonalGaussianLikelihood<float, true>>(m,
                                                      "DiagonalGaussianLikelihood_c8")
      .def(py::init<py::array, py::array, py::array, size_t>())
      .def("apply", &DiagonalGaussianLikelihood<float, true>::apply)
      .def("apply_with_jac", &DiagonalGaussianLikelihood<float, true>::apply_with_jac);
#endif

#ifdef COMPILE_VARCOV_GAUSSIAN_LIKELIHOOD
  py::class_<VariableCovarianceDiagonalGaussianLikelihood<double, false>>(
      m, "VariableCovarianceDiagonalGaussianLikelihood_f8")
      .def(py::init<py::array, py::str, py::str, py::array, size_t>())
      .def("apply", &VariableCovarianceDiagonalGaussianLikelihood<double, false>::apply)
      .def(
          "apply_with_jac",
          &VariableCovarianceDiagonalGaussianLikelihood<double, false>::apply_with_jac);
  py::class_<VariableCovarianceDiagonalGaussianLikelihood<float, false>>(
      m, "VariableCovarianceDiagonalGaussianLikelihood_f4")
      .def(py::init<py::array, py::str, py::str, py::array, size_t>())
      .def("apply", &VariableCovarianceDiagonalGaussianLikelihood<float, false>::apply)
      .def("apply_with_jac",
           &VariableCovarianceDiagonalGaussianLikelihood<float, false>::apply_with_jac);
  py::class_<VariableCovarianceDiagonalGaussianLikelihood<double, true>>(
      m, "VariableCovarianceDiagonalGaussianLikelihood_c16")
      .def(py::init<py::array, py::str, py::str, py::array, size_t>())
      .def("apply", &VariableCovarianceDiagonalGaussianLikelihood<double, true>::apply)
      .def("apply_with_jac",
           &VariableCovarianceDiagonalGaussianLikelihood<double, true>::apply_with_jac);
  py::class_<VariableCovarianceDiagonalGaussianLikelihood<float, true>>(
      m, "VariableCovarianceDiagonalGaussianLikelihood_c8")
      .def(py::init<py::array, py::str, py::str, py::array, size_t>())
      .def("apply", &VariableCovarianceDiagonalGaussianLikelihood<float, true>::apply)
      .def("apply_with_jac",
           &VariableCovarianceDiagonalGaussianLikelihood<float, true>::apply_with_jac);
#endif

#ifdef COMPILE_CALIBRATION_DISTRIBUTOR
  py::class_<CalibrationDistributor>(m, "CalibrationDistributor")
      .def(py::init<py::array, py::array, py::array, py::str, py::str, size_t, size_t,
                    double, size_t>())
      .def("apply", &CalibrationDistributor::apply)
      .def("apply_with_jac", &CalibrationDistributor::apply_with_jac);
#endif

#ifdef COMPILE_CFM
  py::class_<CfmCore>(m, "CfmCore")
      .def(py::init<py::list, py::list, py::str, py::str, double, double, size_t>())
      .def("apply", &CfmCore::apply)
      .def("apply_with_jac", &CfmCore::apply_with_jac);
#endif

  add_linearization<py::array, py::array>(m, "Linearization_field2field");
  add_linearization<py::array, py::dict>(m, "Linearization_field2mfield");
  add_linearization<py::dict, py::array>(m, "Linearization_mfield2field");
  add_linearization<py::dict, py::dict>(m, "Linearization_mfield2mfield");
  add_linearization_with_metric<py::dict>(m, "LinearizationWithMetric_mfield");
  add_linearization_with_metric<py::array>(m, "LinearizationWithMetric_field");
}
