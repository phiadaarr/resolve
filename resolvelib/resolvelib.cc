/*
 *  This file is part of resolvelib.
 *
 *  resolvelib is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  resolvelib is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with nifty_gridder; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/* Copyright (C) 2021-2022 Max-Planck-Society, Philipp Arras
   Authors: Philipp Arras, Jakob Roth */

// Includes related to pybind11
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "ducc0/bindings/pybind_utils.h"
using namespace pybind11::literals;
namespace py = pybind11;
auto None = py::none();
// Includes related to pybind11

#include "shape_helper.h"

using namespace std;

template<typename Tin, typename Tout>
class Linearization {
  public:
    Linearization(const Tout &position_,
                  function<Tout(const Tin  &)> jac_times_,
                  function<Tin (const Tout &)> jac_adjoint_times_)
        : p(position_), jt(jac_times_), jat(jac_adjoint_times_) {}

    Tout jac_times        (const Tin  &inp) const { return jt(inp); }
    Tin  jac_adjoint_times(const Tout &inp) const { return jat(inp); }
    Tout position() { return p; };

  private:
    const Tout p;
    const function<Tout(const Tin  &)> jt;
    const function<Tin (const Tout &)> jat;
};

template<typename Tin, typename Tout>
void add_linearization(py::module_ &msup, const char *name) {
    py::class_<Linearization<Tin, Tout>>(msup, name)
       .def(py::init<const Tout &,
                     function<Tout(const Tin &)>,
                     function<Tin (const Tout &)>
                     >())
       .def("position",          &Linearization<Tin, Tout>::position)
       .def("jac_times",         &Linearization<Tin, Tout>::jac_times)
       .def("jac_adjoint_times", &Linearization<Tin, Tout>::jac_adjoint_times);
}


template<typename T, size_t ndim>
class PolarizationMatrixExponential {
  private:
    const size_t nthreads;
  public:
    PolarizationMatrixExponential(size_t nthreads_=1) : nthreads(nthreads_) {}
    py::array apply(const py::dict &inp_) const {
      // Parse input
      auto I {ducc0::to_cmav<T, ndim>(inp_["I"])},
           Q {ducc0::to_cmav<T, ndim>(inp_["Q"])},
           U {ducc0::to_cmav<T, ndim>(inp_["U"])},
           V {ducc0::to_cmav<T, ndim>(inp_["V"])};
      // /Parse input

      // Instantiate output array
      auto out_ = ducc0::make_Pyarr<T>(combine_shapes(4, I.shape()));
      auto out  = ducc0::to_vmav<T, ndim+1>(out_);
      auto outI = ducc0::subarray<ndim>(out, {{0}, {}, {}, {}, {}}),
           outQ = ducc0::subarray<ndim>(out, {{1}, {}, {}, {}, {}}),
           outU = ducc0::subarray<ndim>(out, {{2}, {}, {}, {}, {}}),
           outV = ducc0::subarray<ndim>(out, {{3}, {}, {}, {}, {}});
      // /Instantiate output array

      ducc0::mav_apply([](const auto &ii, const auto &qq,
                          const auto &uu, const auto &vv,
                          auto &oii, auto &oqq,
                          auto &ouu, auto &ovv
                          ){
              auto pol0{qq*qq + uu*uu + vv*vv};
              auto pol1{-0.5*log(pol0)};
              auto pol{sqrt(pol0)};
              oii = 0.5 * (exp(ii+pol) + exp(ii-pol));
              auto tmp{0.5 * (exp(ii+pol+pol1) - exp(ii-pol+pol1))};
              oqq = tmp * qq;
              ouu = tmp * uu;
              ovv = tmp * vv;
          }, nthreads, I, Q, U, V, outI, outQ, outU, outV);
      return out_;
    }

    Linearization<py::dict,py::array> apply_with_jac(const py::dict &loc_) {

        function<py::array(const py::dict &)> ftimes =
            [&](const py::dict &inp_) {
              // Parse input
              auto I {ducc0::to_cmav<T, ndim>(inp_["I"])},
                   Q {ducc0::to_cmav<T, ndim>(inp_["Q"])},
                   U {ducc0::to_cmav<T, ndim>(inp_["U"])},
                   V {ducc0::to_cmav<T, ndim>(inp_["V"])};
              // /Parse input

              // Instantiate output array
              auto out_ = ducc0::make_Pyarr<T>(combine_shapes(4, I.shape()));
              auto out  = ducc0::to_vmav<T, ndim+1>(out_);
              auto outI = ducc0::subarray<ndim>(out, {{0}, {}, {}, {}, {}}),
                   outQ = ducc0::subarray<ndim>(out, {{1}, {}, {}, {}, {}}),
                   outU = ducc0::subarray<ndim>(out, {{2}, {}, {}, {}, {}}),
                   outV = ducc0::subarray<ndim>(out, {{3}, {}, {}, {}, {}});
              // /Instantiate output array

              return out_;
            };

        function<py::dict(const py::array &)> fadjtimes =
            [&](const py::array &inp_) {
              // Parse input
              auto inp {ducc0::to_cmav<T, ndim+1>(inp_)};
              auto I {ducc0::subarray<ndim>(inp, {{0}, {}, {}, {}, {}})},
                   Q {ducc0::subarray<ndim>(inp, {{1}, {}, {}, {}, {}})},
                   U {ducc0::subarray<ndim>(inp, {{2}, {}, {}, {}, {}})},
                   V {ducc0::subarray<ndim>(inp, {{3}, {}, {}, {}, {}})};
              // /Parse input

              // Instantiate output
              auto outI_ = ducc0::make_Pyarr<T>(I.shape());
              auto outQ_ = ducc0::make_Pyarr<T>(I.shape());
              auto outU_ = ducc0::make_Pyarr<T>(I.shape());
              auto outV_ = ducc0::make_Pyarr<T>(I.shape());

              auto outI {ducc0::to_vmav<T, ndim>(outI_)},
                   outQ {ducc0::to_vmav<T, ndim>(outQ_)},
                   outU {ducc0::to_vmav<T, ndim>(outU_)},
                   outV {ducc0::to_vmav<T, ndim>(outV_)};
              // /Instantiate output

              // Pack into dictionary
              py::dict out_;
              out_["I"] = outI_;
              out_["Q"] = outQ_;
              out_["U"] = outU_;
              out_["V"] = outV_;
              // /Pack into dictionary
              return out_;
            };

        return Linearization<py::dict,py::array>(apply(loc_), ftimes, fadjtimes);
    }
};



PYBIND11_MODULE(resolvelib, m) {
    py::class_<PolarizationMatrixExponential<double, 4>>(m, "PolarizationMatrixExponential")
        .def(py::init<size_t>())
        .def("apply",          &PolarizationMatrixExponential<double, 4>::apply)
        .def("apply_with_jac", &PolarizationMatrixExponential<double, 4>::apply_with_jac);

    add_linearization<py::array, py::array>(m, "Linearization_field2field");
    add_linearization<py::array, py::dict >(m, "Linearization_field2mfield");
    add_linearization<py::dict , py::array>(m, "Linearization_mfield2field");
    add_linearization<py::dict , py::dict >(m, "Linearization_mfield2mfield");
}
