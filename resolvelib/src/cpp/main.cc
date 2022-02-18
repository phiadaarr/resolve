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
      vector<ducc0::slice> slcs(ndim+1);
      slcs[0] = ducc0::slice(0);
      auto outI = ducc0::subarray<ndim>(out, slcs);
      slcs[0] = ducc0::slice(1);
      auto outQ = ducc0::subarray<ndim>(out, slcs);
      slcs[0] = ducc0::slice(2);
      auto outU = ducc0::subarray<ndim>(out, slcs);
      slcs[0] = ducc0::slice(3);
      auto outV = ducc0::subarray<ndim>(out, slcs);
      // /Instantiate output array


      ducc0::mav_apply([](const auto &ii, const auto &qq,
                          const auto &uu, const auto &vv,
                          auto &oii, auto &oqq,
                          auto &ouu, auto &ovv
                          ){
              auto pol{sqrt(qq*qq + uu*uu + vv*vv)};
              auto expii{exp(ii)};
              auto exppol{exp(pol)};
              oii = 0.5 * (expii*exppol + expii/exppol);
              auto tmp{0.5 * ((expii/pol)*exppol - (expii/pol)/exppol)};
              oqq = tmp * qq;
              ouu = tmp * uu;
              ovv = tmp * vv;
          }, nthreads, I, Q, U, V, outI, outQ, outU, outV);
      return out_;
    }

    Linearization<py::dict,py::array> apply_with_jac(const py::dict &loc_) {
        // Parse input
        auto I {ducc0::to_cmav<T, ndim>(loc_["I"])},
             Q {ducc0::to_cmav<T, ndim>(loc_["Q"])},
             U {ducc0::to_cmav<T, ndim>(loc_["U"])},
             V {ducc0::to_cmav<T, ndim>(loc_["V"])};
        // /Parse input

        // Instantiate output array
        auto applied_ = ducc0::make_Pyarr<T>(combine_shapes(4, I.shape()));
        auto applied  = ducc0::to_vmav<T, ndim+1>(applied_);

        vector<ducc0::slice> slcs(ndim+1);
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
          T iq, iu, iv, qq, qu, qv, uq, uu, uv, vq, vu, vv; };

        // Allocate Jacobian
        auto mat = ducc0::vmav<mtx, ndim>(I.shape());
        // /Allocate Jacobian

        // Derive + apply
        ducc0::mav_apply([](const auto &ii , const auto &qq , const auto &uu , const auto &vv ,
                            auto &dii, auto &dqi, auto &dui, auto &dvi,
                            auto &d
                            ){
              auto pol0{qq*qq + uu*uu + vv*vv};
              auto pol{sqrt(pol0)};
              auto expii{exp(ii)};
              auto exppol{exp(pol)};
              auto eplus0{expii*exppol}, eminus0{expii/exppol};
              auto tmp2{0.5 / pol * (eplus0 - eminus0)};
              dii = 0.5 * (eplus0 + eminus0);
              d.iq = tmp2 * qq;
              d.iu = tmp2 * uu;
              d.iv = tmp2 * vv;

              // Tip: never define several variables "auto" together!
              auto eplus{(expii/pol)*exppol};
              auto eminus{(expii/pol)/exppol};
              auto tmp{0.5 * (eplus - eminus)};
              auto tmp3{0.5/pol0 * (eplus*(pol-1.) + eminus*(pol+1.))};

              auto tmpq{tmp3*qq};
              dqi = tmp * qq;
              d.qq = qq * tmpq + tmp;
              d.qu = uu * tmpq;
              d.qv = vv * tmpq;

              auto tmpu{tmp3*uu};
              dui = tmp * uu;
              d.uq = qq * tmpu;
              d.uu = uu * tmpu + tmp;
              d.uv = vv * tmpu;

              auto tmpv{tmp3*vv};
              dvi = tmp * vv;
              d.vq = qq * tmpv;
              d.vu = uu * tmpv;
              d.vv = vv * tmpv + tmp;

            }, nthreads,
               I, Q, U, V,
               appliedI, appliedQ, appliedU, appliedV,
               mat);
        // /Derive + apply

        function<py::array(const py::dict &)> ftimes =
            [nthreads=nthreads,appliedI,appliedQ,appliedU,appliedV,mat](const py::dict &inp_) {
              // Parse input
              auto I {ducc0::to_cmav<T, ndim>(inp_["I"])},
                   Q {ducc0::to_cmav<T, ndim>(inp_["Q"])},
                   U {ducc0::to_cmav<T, ndim>(inp_["U"])},
                   V {ducc0::to_cmav<T, ndim>(inp_["V"])};
              // /Parse input

              // Instantiate output array
              auto out_ = ducc0::make_Pyarr<T>(combine_shapes(4, I.shape()));
              auto out  = ducc0::to_vmav<T, ndim+1>(out_);
              vector<ducc0::slice> slcs(ndim+1);
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
              ducc0::mav_apply([](const auto &dii, const auto &dqi,
                                  const auto &dui, const auto &dvi,
                                  const auto &d,
                                  const auto &ii, const auto &qq,
                                  const auto &uu, const auto &vv,
                                  auto &iiout, auto &qqout,
                                  auto &uuout, auto &vvout){
                    auto ti = dii*ii + d.iq*qq + d.iu*uu + d.iv*vv;
                    auto tq = dqi*ii + d.qq*qq + d.qu*uu + d.qv*vv;
                    auto tu = dui*ii + d.uq*qq + d.uu*uu + d.uv*vv;
                    auto tv = dvi*ii + d.vq*qq + d.vu*uu + d.vv*vv;
                    iiout = ti; qqout = tq; uuout = tu; vvout = tv;
                  }, nthreads, appliedI, appliedQ, appliedU, appliedV,mat,I,Q,U,V,outI,outQ,outU,outV);
              // /Matrix multiplication

              return out_;
            };

        function<py::dict(const py::array &)> fadjtimes =
            [nthreads=nthreads,appliedI,appliedQ,appliedU,appliedV,mat](const py::array &inp_) {
              // Parse input
              auto inp {ducc0::to_cmav<T, ndim+1>(inp_)};
              vector<ducc0::slice> slcs(ndim+1);
              slcs[0] = ducc0::slice(0);
              auto I {ducc0::subarray<ndim>(inp, slcs)};
              slcs[0] = ducc0::slice(1);
              auto Q {ducc0::subarray<ndim>(inp, slcs)};
              slcs[0] = ducc0::slice(2);
              auto U {ducc0::subarray<ndim>(inp, slcs)};
              slcs[0] = ducc0::slice(3);
              auto V {ducc0::subarray<ndim>(inp, slcs)};
              // /Parse input

              // Instantiate output
              py::dict out_;
              out_["I"] = ducc0::make_Pyarr<T>(I.shape());
              out_["Q"] = ducc0::make_Pyarr<T>(I.shape());
              out_["U"] = ducc0::make_Pyarr<T>(I.shape());
              out_["V"] = ducc0::make_Pyarr<T>(I.shape());
              auto outI {ducc0::to_vmav<T, ndim>(out_["I"])},
                   outQ {ducc0::to_vmav<T, ndim>(out_["Q"])},
                   outU {ducc0::to_vmav<T, ndim>(out_["U"])},
                   outV {ducc0::to_vmav<T, ndim>(out_["V"])};
              // /Instantiate output

              // Adjoint matrix multiplication
              ducc0::mav_apply([](const auto &dii, const auto &dqi,
                                  const auto &dui, const auto &dvi,
                                  const auto &d,
                                  const auto &ii, const auto &qq,
                                  const auto &uu, const auto &vv,
                                  auto &iiout, auto &qqout,
                                  auto &uuout, auto &vvout){
                    auto ti = dii *ii + dqi *qq + dui *uu + dvi *vv;
                    auto tq = d.iq*ii + d.qq*qq + d.uq*uu + d.vq*vv;
                    auto tu = d.iu*ii + d.qu*qq + d.uu*uu + d.vu*vv;
                    auto tv = d.iv*ii + d.qv*qq + d.uv*uu + d.vv*vv;
                    iiout = ti; qqout = tq; uuout = tu; vvout = tv;
                  }, nthreads, appliedI, appliedQ, appliedU, appliedV,mat,I,Q,U,V,outI,outQ,outU,outV);
              // /Adjoint matrix multiplication
              return out_;
            };

        return Linearization<py::dict,py::array>(applied_, ftimes, fadjtimes);
    }
};

PYBIND11_MODULE(_cpp, m) {
    m.attr("__name__") = "resolvelib._cpp";
    py::class_<PolarizationMatrixExponential<double, 1>>(m, "PolarizationMatrixExponential1")
        .def(py::init<size_t>())
        .def("apply",          &PolarizationMatrixExponential<double, 1>::apply)
        .def("apply_with_jac", &PolarizationMatrixExponential<double, 1>::apply_with_jac);
    py::class_<PolarizationMatrixExponential<double, 2>>(m, "PolarizationMatrixExponential2")
        .def(py::init<size_t>())
        .def("apply",          &PolarizationMatrixExponential<double, 2>::apply)
        .def("apply_with_jac", &PolarizationMatrixExponential<double, 2>::apply_with_jac);
    py::class_<PolarizationMatrixExponential<double, 3>>(m, "PolarizationMatrixExponential3")
        .def(py::init<size_t>())
        .def("apply",          &PolarizationMatrixExponential<double, 3>::apply)
        .def("apply_with_jac", &PolarizationMatrixExponential<double, 3>::apply_with_jac);
    py::class_<PolarizationMatrixExponential<double, 4>>(m, "PolarizationMatrixExponential4")
        .def(py::init<size_t>())
        .def("apply",          &PolarizationMatrixExponential<double, 4>::apply)
        .def("apply_with_jac", &PolarizationMatrixExponential<double, 4>::apply_with_jac);

    add_linearization<py::array, py::array>(m, "Linearization_field2field");
    add_linearization<py::array, py::dict >(m, "Linearization_field2mfield");
    add_linearization<py::dict , py::array>(m, "Linearization_mfield2field");
    add_linearization<py::dict , py::dict >(m, "Linearization_mfield2mfield");
}
