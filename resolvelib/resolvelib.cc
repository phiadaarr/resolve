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

// Includes related to pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ducc0/bindings/pybind_utils.h"
#include "ducc0/fft/fft.h"
using namespace pybind11::literals;
namespace py = pybind11;
auto None = py::none();
// Includes related to pybind11

#include "shape_helper.h"
using namespace std;

template <typename Tin, typename Tout> class Linearization {
public:
  Linearization(const Tout &position_, function<Tout(const Tin &)> jac_times_,
                function<Tin(const Tout &)> jac_adjoint_times_)
      : p(position_), jt(jac_times_), jat(jac_adjoint_times_) {}

  Tout jac_times(const Tin &inp) const { return jt(inp); }
  Tin jac_adjoint_times(const Tout &inp) const { return jat(inp); }
  Tin apply_metric(const Tin &inp) const { return met(inp); }
  Tout position() { return p; };

protected:
  const Tout p;
  const function<Tout(const Tin &)> jt;
  const function<Tin(const Tout &)> jat;
};

template <typename Tin>
class LinearizationWithMetric : public Linearization<Tin, py::array> {
  using Tout = py::array;

public:
  LinearizationWithMetric(const Tout &position_,
                          function<Tout(const Tin &)> jac_times_,
                          function<Tin(const Tout &)> jac_adjoint_times_,
                          function<Tin(const Tin &)> apply_metric_)
      : Linearization<Tin, Tout>(position_, jac_times_, jac_adjoint_times_),
        met(apply_metric_) {}
  Tin apply_metric(const Tin &inp) const { return met(inp); }

private:
  const function<Tin(const Tin &)> met;
};

template <typename Tin, typename Tout>
void add_linearization(py::module_ &msup, const char *name) {
  py::class_<Linearization<Tin, Tout>>(msup, name)
      .def(py::init<const Tout &, function<Tout(const Tin &)>,
                    function<Tin(const Tout &)>>())
      .def("position", &Linearization<Tin, Tout>::position)
      .def("jac_times", &Linearization<Tin, Tout>::jac_times)
      .def("jac_adjoint_times", &Linearization<Tin, Tout>::jac_adjoint_times);
}
template <typename Tin>
void add_linearization_with_metric(py::module_ &msup, const char *name) {
  using Tout = py::array;
  py::class_<LinearizationWithMetric<Tin>>(msup, name)
      .def(py::init<const Tout &, function<Tout(const Tin &)>,
                    function<Tin(const Tout &)>, function<Tin(const Tin &)>>())
      .def("position", &LinearizationWithMetric<Tin>::position)
      .def("jac_times", &LinearizationWithMetric<Tin>::jac_times)
      .def("jac_adjoint_times",
           &LinearizationWithMetric<Tin>::jac_adjoint_times)
      .def("apply_metric", &LinearizationWithMetric<Tin>::apply_metric);
}

template <typename T, bool complex_mean,
          typename Tmean = conditional_t<complex_mean, complex<T>, T>,
          typename Tacc = long double,
          typename Tacc_cplx = conditional_t<complex_mean, complex<Tacc>, Tacc>>
class DiagonalGaussianLikelihood {
private:
  const size_t nthreads;
  const py::array pymean;
  const py::array pyicov;

public:
  const ducc0::cfmav<Tmean> mean;
  const ducc0::cfmav<T> icov;

  using Tenergy = double;

  DiagonalGaussianLikelihood(const py::array &mean_,
                             const py::array &inverse_covariance_,
                             size_t nthreads_ = 1)
      : nthreads(nthreads_), pymean(mean_), pyicov(inverse_covariance_),
        mean(ducc0::to_cfmav<Tmean>(mean_)),
        icov(ducc0::to_cfmav<T>(inverse_covariance_)) {}

  py::array apply(const py::array &inp_) const {
    auto inp{ducc0::to_cfmav<Tmean>(inp_)};
    Tacc acc{0};
    {
      py::gil_scoped_release release;
      ducc0::mav_apply(
          [&acc](const Tmean &m, const T &ic, const Tmean &l) {
            Tacc_cplx mm(m), ll(l);
            Tacc iicc(ic);
            acc += iicc * norm(ll - mm);
          },
          1, mean, icov, inp); // not parallelized because accumulating
      acc *= 0.5;
    }
    return py::array(py::cast(Tenergy(acc)));
  }

  LinearizationWithMetric<py::array> apply_with_jac(const py::array &loc_) {
    auto loc{ducc0::to_cfmav<Tmean>(loc_)};
    auto gradient{ducc0::vfmav<Tmean>(loc.shape(), ducc0::UNINITIALIZED)};
    Tacc acc{0};

    // value
    ducc0::mav_apply(
        [&acc](const Tmean &m, const T &ic, const Tmean &l) {
          Tacc_cplx mm(m), ll(l);
          Tacc iicc(ic);
          auto tmp1{iicc * norm(ll - mm)};
          acc += tmp1;
        },
        1, mean, icov, loc); // not parallelized because accumulating
    acc *= 0.5;
    auto energy{Tenergy(acc)};
    // /value

    // gradient
    ducc0::mav_apply(
        [](const Tmean &m, const T &ic, const Tmean &l, Tmean &g) {
          auto tmp2{(l - m) * ic};
          g = tmp2;
        },
        nthreads, mean, icov, loc, gradient);
    // /gradient

    // Jacobian
    function<py::array(const py::array &)> ftimes =
        [gradient = gradient](const py::array &inp_) {
          auto inp{ducc0::to_cfmav<Tmean>(inp_)};
          Tacc acc{0};

          ducc0::mav_apply(
              [&acc](const Tmean &i, const Tmean &g) {
                Tacc_cplx ii{i}, gg{g};
                acc += real(ii) * real(gg) + imag(ii) * imag(gg);
              },
              1, inp, gradient); // not parallelized because accumulating
          return py::array(py::cast(Tenergy(acc)));
        };
    function<py::array(const py::array &)> fadjtimes =
        [nthreads = nthreads, gradient = gradient](const py::array &inp_) {
          auto inp{ducc0::to_cfmav<Tenergy>(inp_)};
          auto inpT{T(inp())};
          auto out_{ducc0::make_Pyarr<Tmean>(gradient.shape())};
          auto out{ducc0::to_vfmav<Tmean>(out_)};
          ducc0::mav_apply(
              [inpT = inpT](const Tmean &g, Tmean &o) { o = inpT * g; },
              nthreads, gradient, out);
          return out_;
        };
    // /Jacobian

    // Metric
    function<py::array(const py::array &)> apply_metric =
        [nthreads = nthreads, icov = icov](const py::array &inp_) {
          auto inp{ducc0::to_cfmav<Tmean>(inp_)};
          auto out_{ducc0::make_Pyarr<Tmean>(inp.shape())};
          auto out{ducc0::to_vfmav<Tmean>(out_)};
          ducc0::mav_apply(
              [](const Tmean &i, const T &ic, Tmean &o) { o = i * ic; },
              nthreads, inp, icov, out);
          return out_;
        };
    // /Metric

    return LinearizationWithMetric<py::array>(py::array(py::cast(energy)),
                                              ftimes, fadjtimes, apply_metric);
  }
};

template <typename T, bool complex_mean,
          typename Tmean = conditional_t<complex_mean, complex<T>, T>,
          typename Tacc = long double,
          typename Tacc_cplx = conditional_t<complex_mean, complex<Tacc>, Tacc>>
class VariableCovarianceDiagonalGaussianLikelihood {
private:
  const size_t nthreads;
  const py::array pymean;
  const py::str key_signal;
  const py::str key_log_icov;

public:
  const ducc0::cfmav<Tmean> mean;

  using Tenergy = double;

  VariableCovarianceDiagonalGaussianLikelihood(const py::array &mean_,
                                               const py::str &key_signal_,
                                               const py::str &key_log_icov_,
                                               size_t nthreads_ = 1)
      : nthreads(nthreads_), pymean(mean_), key_signal(key_signal_),
        key_log_icov(key_log_icov_), mean(ducc0::to_cfmav<Tmean>(mean_)) {}

  py::array apply(const py::dict &inp_) const {
    auto signal{ducc0::to_cfmav<Tmean>(inp_[key_signal])};
    auto logicov{ducc0::to_cfmav<T>(inp_[key_log_icov])};
    Tacc acc{0};
    {
      py::gil_scoped_release release;
      ducc0::mav_apply(
          [&acc](const Tmean &m, const T &lic, const Tmean &l) {
            Tacc_cplx mm(m), ll(l);
            Tacc iicc(exp(lic));
            Tacc logiicc(lic);
            if (complex_mean)
              logiicc *= 2;
            acc += iicc * norm(ll - mm) - logiicc;
          },
          1, mean, logicov, signal); // not parallelized because accumulating
      acc *= 0.5;
    }
    return py::array(py::cast(Tenergy(acc)));
  }

  LinearizationWithMetric<py::dict> apply_with_jac(const py::dict &loc_) {
    auto loc_s{ducc0::to_cfmav<Tmean>(loc_[key_signal])};
    auto loc_lic{ducc0::to_cfmav<T>(loc_[key_log_icov])};
    auto grad_s{ducc0::vfmav<Tmean>(loc_s.shape(), ducc0::UNINITIALIZED)};
    auto grad_lic{ducc0::vfmav<T>(loc_lic.shape(), ducc0::UNINITIALIZED)};
    Tacc acc{0};

    // value
    ducc0::mav_apply(
        [&acc](const Tmean &m, const T &lic, const Tmean &l) {
          Tacc_cplx mm(m), ll(l);
          Tacc iicc(exp(lic));
          Tacc logiicc(lic);
          if (complex_mean)
            logiicc *= 2;
          acc += iicc * norm(ll - mm) - logiicc;
        },
        1, mean, loc_lic, loc_s); // not parallelized because accumulating
    acc *= 0.5;
    auto energy{Tenergy(acc)};
    // /value

    // gradient
    // FIXME better variable names for everything...
    ducc0::mav_apply(
        [](const Tmean &m, const Tmean &s, const T &lic, Tmean &gs, T &glic) {
          auto explic{exp(lic)};
          auto tmp2{(s - m) * explic};
          T fct;
          if (complex_mean)
            fct = 2;
          else
            fct = 1;
          T tmp3{T(0.5) * (explic * norm(m - s) - fct)};
          gs = tmp2;
          glic = tmp3;
        },
        nthreads, mean, loc_s, loc_lic, grad_s, grad_lic);
    // /gradient

    // Jacobian
    function<py::array(const py::dict &)> ftimes =
        [grad_s = grad_s, grad_lic = grad_lic, key_signal = key_signal,
         key_log_icov = key_log_icov](const py::dict &inp_) {
          auto inp_s{ducc0::to_cfmav<Tmean>(inp_[key_signal])};
          auto inp_lic{ducc0::to_cfmav<T>(inp_[key_log_icov])};

          Tacc acc{0};

          ducc0::mav_apply(
              [&acc](const Tmean &is, const T &ilic, const Tmean &gs,
                     const T &glic) {
                Tacc_cplx ii{is}, gg{gs};
                Tacc jj{ilic}, hh{glic};
                acc += real(ii) * real(gg) + imag(ii) * imag(gg);
                acc += jj * hh;
              },
              1, inp_s, inp_lic, grad_s,
              grad_lic); // not parallelized because accumulating
          return py::array(py::cast(Tenergy(acc)));
        };

    function<py::dict(const py::array &)> fadjtimes =
        [nthreads = nthreads, grad_s = grad_s, grad_lic = grad_lic,
         key_signal = key_signal,
         key_log_icov = key_log_icov](const py::array &inp_) {
          auto inp{ducc0::to_cfmav<Tenergy>(inp_)};
          auto inpT{T(inp())};
          py::dict out_;
          out_[key_signal] = ducc0::make_Pyarr<Tmean>(grad_s.shape());
          out_[key_log_icov] = ducc0::make_Pyarr<T>(grad_s.shape());
          auto outs{ducc0::to_vfmav<Tmean>(out_[key_signal])};
          auto outlic{ducc0::to_vfmav<T>(out_[key_log_icov])};
          ducc0::mav_apply(
              [inpT = inpT](const Tmean &gs, const T &glic, Tmean &os,
                            T &olic) {
                os = inpT * gs;
                olic = inpT * glic;
              },
              nthreads, grad_s, grad_lic, outs, outlic);
          return out_;
        };
    // /Jacobian

    // Metric
    function<py::dict(const py::dict &)> apply_metric =
        [nthreads = nthreads, key_signal = key_signal,
         key_log_icov = key_log_icov, loc_lic = loc_lic](const py::dict &inp_) {
          auto inp_s{ducc0::to_cfmav<Tmean>(inp_[key_signal])};
          auto inp_lic{ducc0::to_cfmav<T>(inp_[key_log_icov])};

          py::dict out_;
          out_[key_signal] = ducc0::make_Pyarr<Tmean>(inp_s.shape());
          out_[key_log_icov] = ducc0::make_Pyarr<T>(inp_s.shape());
          auto outs{ducc0::to_vfmav<Tmean>(out_[key_signal])};
          auto outlic{ducc0::to_vfmav<T>(out_[key_log_icov])};

          ducc0::mav_apply(
              [](const T &lic, const Tmean &ins, const T &inlic, Tmean &os,
                 T &olic) {
                os = exp(lic) * ins;
                olic = inlic;
                if (!complex_mean)
                  olic *= T(0.5);
              },
              nthreads, loc_lic, inp_s, inp_lic, outs, outlic);
          return out_;
        };
    // /Metric

    return LinearizationWithMetric<py::dict>(py::array(py::cast(energy)),
                                             ftimes, fadjtimes, apply_metric);
  }
};

template <typename T, size_t ndim> class PolarizationMatrixExponential {
private:
  const size_t nthreads;

public:
  PolarizationMatrixExponential(size_t nthreads_ = 1) : nthreads(nthreads_) {}

  py::array apply(const py::dict &inp_) const {
    // Parse input
    auto I{ducc0::to_cmav<T, ndim>(inp_["I"])},
        Q{ducc0::to_cmav<T, ndim>(inp_["Q"])},
        U{ducc0::to_cmav<T, ndim>(inp_["U"])},
        V{ducc0::to_cmav<T, ndim>(inp_["V"])};
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
        [](const auto &ii, const auto &qq, const auto &uu, const auto &vv,
           auto &oii, auto &oqq, auto &ouu, auto &ovv) {
          auto pol{sqrt(qq * qq + uu * uu + vv * vv)};
          auto expii{exp(ii)};
          auto exppol{exp(pol)};
          oii = 0.5 * (expii * exppol + expii / exppol);
          auto tmp{0.5 * ((expii / pol) * exppol - (expii / pol) / exppol)};
          oqq = tmp * qq;
          ouu = tmp * uu;
          ovv = tmp * vv;
        },
        nthreads, I, Q, U, V, outI, outQ, outU, outV);
    return out_;
  }

  Linearization<py::dict, py::array> apply_with_jac(const py::dict &loc_) {
    // Parse input
    auto I{ducc0::to_cmav<T, ndim>(loc_["I"])},
        Q{ducc0::to_cmav<T, ndim>(loc_["Q"])},
        U{ducc0::to_cmav<T, ndim>(loc_["U"])},
        V{ducc0::to_cmav<T, ndim>(loc_["V"])};
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
    auto mat = ducc0::vmav<mtx, ndim>(I.shape());
    // /Allocate Jacobian

    // Derive + apply
    ducc0::mav_apply(
        [](const auto &ii, const auto &qq, const auto &uu, const auto &vv,
           auto &dii, auto &dqi, auto &dui, auto &dvi, auto &d) {
          auto pol0{qq * qq + uu * uu + vv * vv};
          auto pol{sqrt(pol0)};
          auto expii{exp(ii)};
          auto exppol{exp(pol)};
          auto eplus0{expii * exppol}, eminus0{expii / exppol};
          auto tmp2{0.5 / pol * (eplus0 - eminus0)};
          dii = 0.5 * (eplus0 + eminus0);
          d.iq = tmp2 * qq;
          d.iu = tmp2 * uu;
          d.iv = tmp2 * vv;

          // Tip: never define several variables "auto" together!
          auto eplus{(expii / pol) * exppol};
          auto eminus{(expii / pol) / exppol};
          auto tmp{0.5 * (eplus - eminus)};
          auto tmp3{0.5 / pol0 * (eplus * (pol - 1.) + eminus * (pol + 1.))};

          auto tmpq{tmp3 * qq};
          dqi = tmp * qq;
          d.qq = qq * tmpq + tmp;
          d.qu = uu * tmpq;
          d.qv = vv * tmpq;

          auto tmpu{tmp3 * uu};
          dui = tmp * uu;
          d.uq = qq * tmpu;
          d.uu = uu * tmpu + tmp;
          d.uv = vv * tmpu;

          auto tmpv{tmp3 * vv};
          dvi = tmp * vv;
          d.vq = qq * tmpv;
          d.vu = uu * tmpv;
          d.vv = vv * tmpv + tmp;
        },
        nthreads, I, Q, U, V, appliedI, appliedQ, appliedU, appliedV, mat);
    // /Derive + apply

    function<py::array(const py::dict &)> ftimes =
        [nthreads = nthreads, appliedI, appliedQ, appliedU, appliedV,
         mat](const py::dict &inp_) {
          // Parse input
          auto I{ducc0::to_cmav<T, ndim>(inp_["I"])},
              Q{ducc0::to_cmav<T, ndim>(inp_["Q"])},
              U{ducc0::to_cmav<T, ndim>(inp_["U"])},
              V{ducc0::to_cmav<T, ndim>(inp_["V"])};
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
              [](const auto &dii, const auto &dqi, const auto &dui,
                 const auto &dvi, const auto &d, const auto &ii, const auto &qq,
                 const auto &uu, const auto &vv, auto &iiout, auto &qqout,
                 auto &uuout, auto &vvout) {
                auto ti = dii * ii + d.iq * qq + d.iu * uu + d.iv * vv;
                auto tq = dqi * ii + d.qq * qq + d.qu * uu + d.qv * vv;
                auto tu = dui * ii + d.uq * qq + d.uu * uu + d.uv * vv;
                auto tv = dvi * ii + d.vq * qq + d.vu * uu + d.vv * vv;
                iiout = ti;
                qqout = tq;
                uuout = tu;
                vvout = tv;
              },
              nthreads, appliedI, appliedQ, appliedU, appliedV, mat, I, Q, U, V,
              outI, outQ, outU, outV);
          // /Matrix multiplication

          return out_;
        };

    function<py::dict(const py::array &)> fadjtimes =
        [nthreads = nthreads, appliedI, appliedQ, appliedU, appliedV,
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
              [](const auto &dii, const auto &dqi, const auto &dui,
                 const auto &dvi, const auto &d, const auto &ii, const auto &qq,
                 const auto &uu, const auto &vv, auto &iiout, auto &qqout,
                 auto &uuout, auto &vvout) {
                auto ti = dii * ii + dqi * qq + dui * uu + dvi * vv;
                auto tq = d.iq * ii + d.qq * qq + d.uq * uu + d.vq * vv;
                auto tu = d.iu * ii + d.qu * qq + d.uu * uu + d.vu * vv;
                auto tv = d.iv * ii + d.qv * qq + d.uv * uu + d.vv * vv;
                iiout = ti;
                qqout = tq;
                uuout = tu;
                vvout = tv;
              },
              nthreads, appliedI, appliedQ, appliedU, appliedV, mat, I, Q, U, V,
              outI, outQ, outU, outV);
          // /Adjoint matrix multiplication
          return out_;
        };

    return Linearization<py::dict, py::array>(applied_, ftimes, fadjtimes);
  }
};

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

  size_t nrows() const {
    return ducc0::to_cmav<int, 1>(antenna_indices0).shape()[0];
  } // FIXME Use unsigned for indices

public:
  CalibrationDistributor(const py::array &antenna_indices0_,
                         const py::array &antenna_indices1_,
                         const py::array &time_,
                         const py::str &key_logamplitude_,
                         const py::str &key_phase_, size_t nfreqs_,
                         size_t ntime_, double dt_, size_t nthreads_)
      : key_logamplitude(key_logamplitude_), key_phase(key_phase_),
        antenna_indices0(antenna_indices0_),
        antenna_indices1(antenna_indices1_), time(time_), nfreqs(nfreqs_),
        ntime(ntime_), dt(dt_), nthreads(nthreads_) {}

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
              const auto loggain = complex<double>(
                  logampl(i0, a0(i1), tindex, i2) +
                      logampl(i0, a1(i1), tindex, i2),
                  ph(i0, a0(i1), tindex, i2) - ph(i0, a1(i1), tindex, i2));
              return loggain;
            };
            const auto diff{frac - double(tind0)};
            const auto loggain{(1 - diff) * getloggain(tind0) +
                               diff * getloggain(tind1)};
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
      const auto inp_logampl =
          ducc0::to_cmav<double, 4>(inp_[key_logamplitude]);
      const auto inp_ph = ducc0::to_cmav<double, 4>(inp_[key_phase]);
      // /Parse input

      // Instantiate output array
      auto npol{inp_ph.shape()[0]};
      auto out_ = ducc0::make_Pyarr<complex<double>>({npol, nrows(), nfreqs});
      auto out = ducc0::to_vmav<complex<double>, 3>(out_);
      // /Instantiate output array

      for (size_t i0 = 0; i0 < out.shape()[0]; ++i0)
        ducc0::execParallel(
            out.shape()[1], nthreads, [&](size_t lo, size_t hi) {
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
                           (inp_logampl(i0, a0(i1), tindex, i2) +
                            inp_logampl(i0, a1(i1), tindex, i2) +
                            complex<double>{0, 1} *
                                (inp_ph(i0, a0(i1), tindex, i2) -
                                 inp_ph(i0, a1(i1), tindex, i2)));
                  };

                  const complex<double> tmp0{gettmp(tind0)};
                  const complex<double> tmp1{gettmp(tind1)};

                  auto diff{frac - double(tind0)};
                  auto tmp{(1 - diff) * tmp0 + diff * tmp1};

                  out(i0, i1, i2) = tmp;
                }
            });
      return out_;
    };

    function<py::dict(const py::array &)> fadjtimes =
        [=](const py::array &inp_) {
          // Parse input
          auto inp{ducc0::to_cmav<complex<double>, 3>(inp_)};
          // /Parse input

          // Instantiate output
          py::dict out_;
          out_[key_logamplitude] = ducc0::make_Pyarr<double>(inp_shape);
          out_[key_phase] = ducc0::make_Pyarr<double>(inp_shape);
          auto logampl{ducc0::to_vmav<double, 4>(out_[key_logamplitude])};
          auto logph{ducc0::to_vmav<double, 4>(out_[key_phase])};
          ducc0::mav_apply([](double &inp) { inp = 0; }, 1, logampl);
          ducc0::mav_apply([](double &inp) { inp = 0; }, 1, logph);

          for (size_t i0 = 0; i0 < inp.shape()[0]; ++i0)
            ducc0::execParallel(
                inp.shape()[1], nthreads, [&](size_t lo, size_t hi) {
                  for (size_t i1 = lo; i1 < hi; ++i1)
                    for (size_t i2 = 0; i2 < inp.shape()[2]; ++i2) {
                      const double frac{t(i1) / dt};
                      const auto tind0 = size_t(floor(frac));
                      const size_t tind1{tind0 + 1};
                      MR_assert(tind0 < ntime, "time outside region");
                      MR_assert(tind1 < ntime, "time outside region");

                      auto diff{frac - double(tind0)};

                      const auto tmp{conj(applied(i0, i1, i2)) *
                                     inp(i0, i1, i2)};
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
                });
          return out_;
        };

    return Linearization<py::dict, py::array>(applied_, ftimes, fadjtimes);
  }
};

class CfmCore {
private:
  const py::list amplitude_keys;
  const py::list pindices;
  const py::str key_xi;
  const py::str key_azm;
  const double offset_mean;
  size_t nthreads;

  using shape_t = vector<size_t>;

  ducc0::cfmav<int64_t>
  pindex(const size_t &index) const { // FIXME @mtr is this type correct? This
                                      // is at least what python gives me
    return ducc0::to_cfmav<int64_t>(pindices[index]);
  }

public:
  CfmCore(const py::list &pindices_, const py::list &amplitude_keys_,
          const py::str &key_xi_, const py::str &key_azm_,
          const double &offset_mean_, const size_t nthreads_)
      : amplitude_keys(amplitude_keys_), pindices(pindices_), key_xi(key_xi_),
        key_azm(key_azm_), offset_mean(offset_mean_), nthreads(nthreads_) {}

  py::array apply(const py::dict &inp_) const {
    const auto inp_xi = ducc0::to_cfmav<double>(inp_[key_xi]);
    const auto inp_azm = ducc0::to_cfmav<double>(inp_[key_azm]);
    const auto inp_pspec0{ducc0::to_cmav<double, 2>(inp_[amplitude_keys[0]])};
    const auto inp_pspec1{ducc0::to_cmav<double, 2>(inp_[amplitude_keys[1]])};

    auto out_ = ducc0::make_Pyarr<double>(inp_xi.shape());
    auto out = ducc0::to_vfmav<double>(out_);

    // xi and Power distributor
    const auto p0{pindex(0)};
    const auto p1{pindex(1)};
    ducc0::mav_apply_with_index(
        [&](double &oo, const double &xi, const shape_t &inds) {
          const int64_t ind0{p0(inds[1])};
          const int64_t ind1{p1(inds[2])};
          const double foo{inp_pspec0(inds[0], ind0) *
                           inp_pspec1(inds[0], ind1) * inp_azm(inds[0]) * xi};
          oo = foo;
        },
        nthreads, out, inp_xi);
    // /Power distributor

    // Offset mean
    vector<ducc0::slice> slcs(3);
    for (size_t i = 0; i < inp_xi.shape(0); ++i)
      out(i, 0, 0) += offset_mean;
    // /Offset mean

    // FFT
    ducc0::r2r_genuine_hartley(out, out, {1}, 1., nthreads);
    ducc0::r2r_genuine_hartley(out, out, {2}, 1., nthreads);
    // /FFT

    return out_;
  }

  Linearization<py::dict, py::array> apply_with_jac(const py::dict &inp_) {
    const auto inp_xi = ducc0::to_cfmav<double>(inp_[key_xi]);
    const auto inp_azm = ducc0::to_cfmav<double>(inp_[key_azm]);
    const auto inp_pspec0{ducc0::to_cmav<double, 2>(inp_[amplitude_keys[0]])};
    const auto inp_pspec1{ducc0::to_cmav<double, 2>(inp_[amplitude_keys[1]])};

    auto out_ = ducc0::make_Pyarr<double>(inp_xi.shape());
    auto out = ducc0::to_vfmav<double>(out_);

    // xi and Power distributor
    const auto p0{pindex(0)};
    const auto p1{pindex(1)};
    ducc0::mav_apply_with_index(
        [&](double &oo, const double &xi, const shape_t &inds) {
          const int64_t ind0{p0(inds[1])}, ind1{p1(inds[2])};
          const double fac0{inp_pspec0(inds[0], ind0)},
              fac1{inp_pspec1(inds[0], ind1)}, fac2{inp_azm(inds[0])}, fac3{xi};
          const double foo{fac0 * fac1 * fac2 * fac3};
          oo = foo;
        },
        nthreads, out, inp_xi);
    // /Power distributor

    // Offset mean
    vector<ducc0::slice> slcs(3);
    for (size_t i = 0; i < inp_xi.shape(0); ++i)
      out(i, 0, 0) += offset_mean;
    // /Offset mean

    // FFT
    ducc0::r2r_genuine_hartley(out, out, {1}, 1., nthreads);
    ducc0::r2r_genuine_hartley(out, out, {2}, 1., nthreads);
    // /FFT

    function<py::array(const py::dict &)> ftimes =
        [=](const py::dict &tangent_) {
          auto out_ = ducc0::make_Pyarr<double>(inp_xi.shape());
          auto out = ducc0::to_vfmav<double>(out_);
          const auto tangent_xi = ducc0::to_cfmav<double>(tangent_[key_xi]);
          const auto tangent_azm = ducc0::to_cfmav<double>(tangent_[key_azm]);
          const auto tangent_pspec0{
              ducc0::to_cmav<double, 2>(tangent_[amplitude_keys[0]])};
          const auto tangent_pspec1{
              ducc0::to_cmav<double, 2>(tangent_[amplitude_keys[1]])};

          // xi and Power distributor
          ducc0::mav_apply_with_index(
              [&](double &oo, const double &xi0, const double &dxi,
                  const shape_t &inds) {
                const int64_t ind0{p0(inds[1])}, ind1{p1(inds[2])};
                const double fac0{inp_pspec0(inds[0], ind0)},
                    fac1{inp_pspec1(inds[0], ind1)}, fac2{inp_azm(inds[0])},
                    fac3{xi0};
                const double d0{tangent_pspec0(inds[0], ind0)},
                    d1{tangent_pspec1(inds[0], ind1)}, d2{tangent_azm(inds[0])},
                    d3{dxi};
                const double foo{
                    d0 * fac1 * fac2 * fac3 + fac0 * d1 * fac2 * fac3 +
                    fac0 * fac1 * d2 * fac3 + fac0 * fac1 * fac2 * d3};
                oo = foo;
              },
              nthreads, out, inp_xi, tangent_xi);
          // /Power distributor
          // FFT
          ducc0::r2r_genuine_hartley(out, out, {1}, 1., nthreads);
          ducc0::r2r_genuine_hartley(out, out, {2}, 1., nthreads);
          // /FFT
          return out_;
        };

    function<py::dict(const py::array &)> fadjtimes =
        [=](const py::array &cotangent_) {
          auto inpcopy = inp_; // keep inp_ alive to avoid dangling references
          const auto cotangent = ducc0::to_cfmav<double>(cotangent_);
          py::dict out_;
          out_[key_xi] = ducc0::make_Pyarr<double>(inp_xi.shape());
          out_[key_azm] = ducc0::make_Pyarr<double>(inp_azm.shape());
          out_[amplitude_keys[0]] =
              ducc0::make_Pyarr<double>(inp_pspec0.shape());
          out_[amplitude_keys[1]] =
              ducc0::make_Pyarr<double>(inp_pspec1.shape());
          auto out_xi = ducc0::to_vfmav<double>(out_[key_xi]);
          auto out_azm = ducc0::to_vfmav<double>(out_[key_azm]);
          auto out_pspec0 = ducc0::to_vfmav<double>(out_[amplitude_keys[0]]);
          auto out_pspec1 = ducc0::to_vfmav<double>(out_[amplitude_keys[1]]);

          // ducc0::mav_apply([](double &inp) { inp = 0; }, nthreads, out_xi);
          ducc0::mav_apply([](double &inp) { inp = 0; }, nthreads, out_azm);
          ducc0::mav_apply([](double &inp) { inp = 0; }, nthreads, out_pspec0);
          ducc0::mav_apply([](double &inp) { inp = 0; }, nthreads, out_pspec1);

          // FFT
          ducc0::r2r_genuine_hartley(cotangent, out_xi, {2}, 1., nthreads);
          ducc0::r2r_genuine_hartley(out_xi, out_xi, {1}, 1., nthreads);

          // xi and Power distributor
          ducc0::mav_apply_with_index(
              [&](const double &xi0, double &dxi, const shape_t &inds) {
                const int64_t ind0{p0(inds[1])}, ind1{p1(inds[2])};
                const double fac0{inp_pspec0(inds[0], ind0)},
                    fac1{inp_pspec1(inds[0], ind1)}, fac2{inp_azm(inds[0])},
                    fac3{xi0};
                out_pspec0(inds[0], ind0) += fac1 * fac2 * fac3 * dxi;
                out_pspec1(inds[0], ind1) += fac0 * fac2 * fac3 * dxi;
                out_azm(inds[0]) += fac0 * fac1 * fac3 * dxi;
                dxi *= fac0 * fac1 * fac2;
              },
              1, inp_xi, out_xi);

          return out_;
        };

    return Linearization<py::dict, py::array>(out_, ftimes, fadjtimes);
  }
};

PYBIND11_MODULE(resolvelib, m) {

  m.attr("__name__") = "resolvelib";

  py::class_<PolarizationMatrixExponential<double, 1>>(
      m, "PolarizationMatrixExponential1")
      .def(py::init<size_t>())
      .def("apply", &PolarizationMatrixExponential<double, 1>::apply)
      .def("apply_with_jac",
           &PolarizationMatrixExponential<double, 1>::apply_with_jac);
  py::class_<PolarizationMatrixExponential<double, 2>>(
      m, "PolarizationMatrixExponential2")
      .def(py::init<size_t>())
      .def("apply", &PolarizationMatrixExponential<double, 2>::apply)
      .def("apply_with_jac",
           &PolarizationMatrixExponential<double, 2>::apply_with_jac);
  py::class_<PolarizationMatrixExponential<double, 3>>(
      m, "PolarizationMatrixExponential3")
      .def(py::init<size_t>())
      .def("apply", &PolarizationMatrixExponential<double, 3>::apply)
      .def("apply_with_jac",
           &PolarizationMatrixExponential<double, 3>::apply_with_jac);
  py::class_<PolarizationMatrixExponential<double, 4>>(
      m, "PolarizationMatrixExponential4")
      .def(py::init<size_t>())
      .def("apply", &PolarizationMatrixExponential<double, 4>::apply)
      .def("apply_with_jac",
           &PolarizationMatrixExponential<double, 4>::apply_with_jac);

  py::class_<DiagonalGaussianLikelihood<double, false>>(
      m, "DiagonalGaussianLikelihood_f8")
      .def(py::init<py::array, py::array, size_t>())
      .def("apply", &DiagonalGaussianLikelihood<double, false>::apply)
      .def("apply_with_jac",
           &DiagonalGaussianLikelihood<double, false>::apply_with_jac);
  py::class_<DiagonalGaussianLikelihood<float, false>>(
      m, "DiagonalGaussianLikelihood_f4")
      .def(py::init<py::array, py::array, size_t>())
      .def("apply", &DiagonalGaussianLikelihood<float, false>::apply)
      .def("apply_with_jac",
           &DiagonalGaussianLikelihood<float, false>::apply_with_jac);
  py::class_<DiagonalGaussianLikelihood<double, true>>(
      m, "DiagonalGaussianLikelihood_c16")
      .def(py::init<py::array, py::array, size_t>())
      .def("apply", &DiagonalGaussianLikelihood<double, true>::apply)
      .def("apply_with_jac",
           &DiagonalGaussianLikelihood<double, true>::apply_with_jac);
  py::class_<DiagonalGaussianLikelihood<float, true>>(
      m, "DiagonalGaussianLikelihood_c8")
      .def(py::init<py::array, py::array, size_t>())
      .def("apply", &DiagonalGaussianLikelihood<float, true>::apply)
      .def("apply_with_jac",
           &DiagonalGaussianLikelihood<float, true>::apply_with_jac);

  py::class_<VariableCovarianceDiagonalGaussianLikelihood<double, false>>(
      m, "VariableCovarianceDiagonalGaussianLikelihood_f8")
      .def(py::init<py::array, py::str, py::str, size_t>())
      .def("apply",
           &VariableCovarianceDiagonalGaussianLikelihood<double, false>::apply)
      .def("apply_with_jac", &VariableCovarianceDiagonalGaussianLikelihood<
                                 double, false>::apply_with_jac);
  py::class_<VariableCovarianceDiagonalGaussianLikelihood<float, false>>(
      m, "VariableCovarianceDiagonalGaussianLikelihood_f4")
      .def(py::init<py::array, py::str, py::str, size_t>())
      .def("apply",
           &VariableCovarianceDiagonalGaussianLikelihood<float, false>::apply)
      .def("apply_with_jac", &VariableCovarianceDiagonalGaussianLikelihood<
                                 float, false>::apply_with_jac);
  py::class_<VariableCovarianceDiagonalGaussianLikelihood<double, true>>(
      m, "VariableCovarianceDiagonalGaussianLikelihood_c16")
      .def(py::init<py::array, py::str, py::str, size_t>())
      .def("apply",
           &VariableCovarianceDiagonalGaussianLikelihood<double, true>::apply)
      .def("apply_with_jac",
           &VariableCovarianceDiagonalGaussianLikelihood<double,
                                                         true>::apply_with_jac);
  py::class_<VariableCovarianceDiagonalGaussianLikelihood<float, true>>(
      m, "VariableCovarianceDiagonalGaussianLikelihood_c8")
      .def(py::init<py::array, py::str, py::str, size_t>())
      .def("apply",
           &VariableCovarianceDiagonalGaussianLikelihood<float, true>::apply)
      .def("apply_with_jac",
           &VariableCovarianceDiagonalGaussianLikelihood<float,
                                                         true>::apply_with_jac);

  py::class_<CalibrationDistributor>(m, "CalibrationDistributor")
      .def(py::init<py::array, py::array, py::array, py::str, py::str, size_t,
                    size_t, double, size_t>())
      .def("apply", &CalibrationDistributor::apply)
      .def("apply_with_jac", &CalibrationDistributor::apply_with_jac);

  py::class_<CfmCore>(m, "CfmCore")
      .def(py::init<py::list, py::list, py::str, py::str, double, size_t>())
      .def("apply", &CfmCore::apply)
      .def("apply_with_jac", &CfmCore::apply_with_jac);

  add_linearization<py::array, py::array>(m, "Linearization_field2field");
  add_linearization<py::array, py::dict>(m, "Linearization_field2mfield");
  add_linearization<py::dict, py::array>(m, "Linearization_mfield2field");
  add_linearization<py::dict, py::dict>(m, "Linearization_mfield2mfield");
  add_linearization_with_metric<py::dict>(m, "LinearizationWithMetric_mfield");
  add_linearization_with_metric<py::array>(m, "LinearizationWithMetric_field");
}
