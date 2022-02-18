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

#define NEW_VERSION


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
#ifdef NEW_VERSION
              auto pol{sqrt(qq*qq + uu*uu + vv*vv)};
              auto expii{exp(ii)};
              auto exppol{exp(pol)};
              oii = 0.5 * (expii*exppol + expii/exppol);
              auto tmp{0.5 * ((expii/pol)*exppol - (expii/pol)/exppol)};
#else
              auto pol0{qq*qq + uu*uu + vv*vv};
              auto pol{sqrt(pol0)};
              auto pol1{log(pol)};
              oii = 0.5 * (exp(ii+pol) + exp(ii-pol));
              auto diff{ii-pol1};
              auto tmp{0.5 * (exp(diff+pol) - exp(diff-pol))};
#endif
              oqq = tmp * qq;
              ouu = tmp * uu;
              ovv = tmp * vv;
          }, nthreads, I, Q, U, V, outI, outQ, outU, outV);
      return out_;
    }

#ifdef NEW_VERSION
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
        auto appliedI = ducc0::subarray<ndim>(applied, {{0}, {}, {}, {}, {}}),
             appliedQ = ducc0::subarray<ndim>(applied, {{1}, {}, {}, {}, {}}),
             appliedU = ducc0::subarray<ndim>(applied, {{2}, {}, {}, {}, {}}),
             appliedV = ducc0::subarray<ndim>(applied, {{3}, {}, {}, {}, {}});
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
              auto outI = ducc0::subarray<ndim>(out, {{0}, {}, {}, {}, {}}),
                   outQ = ducc0::subarray<ndim>(out, {{1}, {}, {}, {}, {}}),
                   outU = ducc0::subarray<ndim>(out, {{2}, {}, {}, {}, {}}),
                   outV = ducc0::subarray<ndim>(out, {{3}, {}, {}, {}, {}});
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
              auto I {ducc0::subarray<ndim>(inp, {{0}, {}, {}, {}, {}})},
                   Q {ducc0::subarray<ndim>(inp, {{1}, {}, {}, {}, {}})},
                   U {ducc0::subarray<ndim>(inp, {{2}, {}, {}, {}, {}})},
                   V {ducc0::subarray<ndim>(inp, {{3}, {}, {}, {}, {}})};
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
#else
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
        auto appliedI = ducc0::subarray<ndim>(applied, {{0}, {}, {}, {}, {}}),
             appliedQ = ducc0::subarray<ndim>(applied, {{1}, {}, {}, {}, {}}),
             appliedU = ducc0::subarray<ndim>(applied, {{2}, {}, {}, {}, {}}),
             appliedV = ducc0::subarray<ndim>(applied, {{3}, {}, {}, {}, {}});
        // /Instantiate output array

        // Allocate Jacobian
        auto shp{I.shape()};
        auto d00 = appliedI;
        auto d10 = appliedQ;
        auto d20 = appliedU;
        auto d30 = appliedV;
        auto d01 = ducc0::vmav<T, ndim>(shp);
        auto d02 = ducc0::vmav<T, ndim>(shp);
        auto d03 = ducc0::vmav<T, ndim>(shp);
        auto d11 = ducc0::vmav<T, ndim>(shp);
        auto d12 = ducc0::vmav<T, ndim>(shp);
        auto d13 = ducc0::vmav<T, ndim>(shp);
        auto d21 = ducc0::vmav<T, ndim>(shp);
        auto d22 = ducc0::vmav<T, ndim>(shp);
        auto d23 = ducc0::vmav<T, ndim>(shp);
        auto d31 = ducc0::vmav<T, ndim>(shp);
        auto d32 = ducc0::vmav<T, ndim>(shp);
        auto d33 = ducc0::vmav<T, ndim>(shp);
        // /Allocate Jacobian

        // Derive + apply
        ducc0::mav_apply([](const auto &ii , const auto &qq , const auto &uu , const auto &vv ,
                            auto &oii, auto &oqq, auto &ouu, auto &ovv,
                            auto &oiiqq, auto &oiiuu, auto &oiivv,
                            auto &oqqqq, auto &oqquu, auto &oqqvv,
                            auto &ouuqq, auto &ouuuu, auto &ouuvv,
                            auto &ovvqq, auto &ovvuu, auto &ovvvv

                            ){
              auto pol0{qq*qq + uu*uu + vv*vv};
              auto pol{sqrt(pol0)};
              auto eplus0{exp(ii+pol)}, eminus0{exp(ii-pol)};
              auto tmp2{0.5 / pol * (eplus0 - eminus0)};
              oii = 0.5 * (eplus0 + eminus0);
              oiiqq = tmp2 * qq;
              oiiuu = tmp2 * uu;
              oiivv = tmp2 * vv;

              auto diff{ii-log(pol)};
              auto eplus{exp(diff+pol)}, eminus{exp(diff-pol)};
              auto tmp{0.5 * (eplus - eminus)};
              auto tmp3{0.5/pol0 * (eplus*(pol-1.) + eminus*(pol+1.))};

              auto tmpq{tmp3*qq};
              oqq = tmp * qq;
              oqqqq = qq * tmpq + tmp;
              oqquu = uu * tmpq;
              oqqvv = vv * tmpq;

              auto tmpu{tmp3*uu};
              ouu = tmp * uu;
              ouuqq = qq * tmpu;
              ouuuu = uu * tmpu + tmp;
              ouuvv = vv * tmpu;

              auto tmpv{tmp3*vv};
              ovv = tmp * vv;
              ovvqq = qq * tmpv;
              ovvuu = uu * tmpv;
              ovvvv = vv * tmpv + tmp;

            }, nthreads,
               I, Q, U, V,
               appliedI, appliedQ, appliedU, appliedV,
               d01, d02, d03,
               d11, d12, d13,
               d21, d22, d23,
               d31, d32, d33);
        // /Derive + apply

        function<py::array(const py::dict &)> ftimes =
            [=](const py::dict &inp_) {
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

              // Matrix multiplication
              ducc0::mav_apply([](const auto &ii0, const auto &qq0,
                                  const auto &uu0, const auto &vv0,
                                  const auto &ii,  const auto &qq,
                                  const auto &uu,  const auto &vv,
                                  auto &out){
                    out = ii0*ii + qq0*qq + uu0*uu + vv0*vv;
                  }, nthreads, d00, d01, d02, d03, I, Q, U, V, outI);
              ducc0::mav_apply([](const auto &ii0, const auto &qq0,
                                  const auto &uu0, const auto &vv0,
                                  const auto &ii,  const auto &qq,
                                  const auto &uu,  const auto &vv,
                                  auto &out){
                    out = ii0*ii + qq0*qq + uu0*uu + vv0*vv;
                  }, nthreads, d10, d11, d12, d13, I, Q, U, V, outQ);
              ducc0::mav_apply([](const auto &ii0, const auto &qq0,
                                  const auto &uu0, const auto &vv0,
                                  const auto &ii,  const auto &qq,
                                  const auto &uu,  const auto &vv,
                                  auto &out){
                    out = ii0*ii + qq0*qq + uu0*uu + vv0*vv;
                  }, nthreads, d20, d21, d22, d23, I, Q, U, V, outU);
              ducc0::mav_apply([](const auto &ii0, const auto &qq0,
                                  const auto &uu0, const auto &vv0,
                                  const auto &ii,  const auto &qq,
                                  const auto &uu,  const auto &vv,
                                  auto &out){
                    out = ii0*ii + qq0*qq + uu0*uu + vv0*vv;
                  }, nthreads, d30, d31, d32, d33, I, Q, U, V, outV);
              // /Matrix multiplication

              return out_;
            };

        function<py::dict(const py::array &)> fadjtimes =
            [=](const py::array &inp_) {
              // Parse input
              auto inp {ducc0::to_cmav<T, ndim+1>(inp_)};
              auto I {ducc0::subarray<ndim>(inp, {{0}, {}, {}, {}, {}})},
                   Q {ducc0::subarray<ndim>(inp, {{1}, {}, {}, {}, {}})},
                   U {ducc0::subarray<ndim>(inp, {{2}, {}, {}, {}, {}})},
                   V {ducc0::subarray<ndim>(inp, {{3}, {}, {}, {}, {}})};
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
              ducc0::mav_apply([](const auto &ii0, const auto &qq0,
                                  const auto &uu0, const auto &vv0,
                                  const auto &ii,  const auto &qq,
                                  const auto &uu,  const auto &vv,
                                  auto &out){
                    out = ii0*ii + qq0*qq + uu0*uu + vv0*vv;
                  }, nthreads, d00, d10, d20, d30, I, Q, U, V, outI);
              ducc0::mav_apply([](const auto &ii0, const auto &qq0,
                                  const auto &uu0, const auto &vv0,
                                  const auto &ii,  const auto &qq,
                                  const auto &uu,  const auto &vv,
                                  auto &out){
                    out = ii0*ii + qq0*qq + uu0*uu + vv0*vv;
                  }, nthreads, d01, d11, d21, d31, I, Q, U, V, outQ);
              ducc0::mav_apply([](const auto &ii0, const auto &qq0,
                                  const auto &uu0, const auto &vv0,
                                  const auto &ii,  const auto &qq,
                                  const auto &uu,  const auto &vv,
                                  auto &out){
                    out = ii0*ii + qq0*qq + uu0*uu + vv0*vv;
                  }, nthreads, d02, d12, d22, d32, I, Q, U, V, outU);
              ducc0::mav_apply([](const auto &ii0, const auto &qq0,
                                  const auto &uu0, const auto &vv0,
                                  const auto &ii,  const auto &qq,
                                  const auto &uu,  const auto &vv,
                                  auto &out){
                    out = ii0*ii + qq0*qq + uu0*uu + vv0*vv;
                  }, nthreads, d03, d13, d23, d33, I, Q, U, V, outV);
              // /Adjoint matrix multiplication
              return out_;
            };

        return Linearization<py::dict,py::array>(applied_, ftimes, fadjtimes);
    }
#endif
};



PYBIND11_MODULE(_cpp, m) {
    m.attr("__name__") = "resolvelib._cpp";
    py::class_<PolarizationMatrixExponential<double, 4>>(m, "PolarizationMatrixExponential")
        .def(py::init<size_t>())
        .def("apply",          &PolarizationMatrixExponential<double, 4>::apply)
        .def("apply_with_jac", &PolarizationMatrixExponential<double, 4>::apply_with_jac);

    add_linearization<py::array, py::array>(m, "Linearization_field2field");
    add_linearization<py::array, py::dict >(m, "Linearization_field2mfield");
    add_linearization<py::dict , py::array>(m, "Linearization_mfield2field");
    add_linearization<py::dict , py::dict >(m, "Linearization_mfield2mfield");
}
