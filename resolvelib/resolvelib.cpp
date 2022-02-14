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
#include "ducc0/infra/threading.h"
using namespace pybind11::literals;
namespace py = pybind11;
auto None = py::none();
// Includes related to pybind11

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


class PolarizationMatrixExponential {
  public:
    PolarizationMatrixExponential() {}
    py::array apply(const py::array &inp) const {
      // Assume ordering: I, Q, U, V
        MR_fail("Not implemented yet");
        return inp;
    }

    Linearization<py::array,py::array> apply_with_jac(const py::array &inp) {
        function<py::array(const py::array &)> f =
          [=](const py::array &inp2) {
            MR_fail("Not implemented yet");
            return inp2;
        };
        function<py::array(const py::array &)> ftimes =
            [=](const py::array &inp2) { return f(inp2); };
        function<py::array(const py::array &)> fadjtimes =
            [=](const py::array &inp2) { return f(inp2); };

        return Linearization<py::array,py::array> {apply(inp), ftimes, fadjtimes};
    }
};


template<typename Tin, typename Tout>
void add_linearization(py::module_ &msup, const string &name) {
    py::class_<Linearization<Tin, Tout>>(msup, name)
       .def(py::init<const Tout &,
                     function<Tout(const Tin &)>,
                     function<Tin (const Tout &)>
                     >())
       .def("position",          &Linearization<Tin, Tout>::position)
       .def("jac_times",         &Linearization<Tin, Tout>::jac_times)
       .def("jac_adjoint_times", &Linearization<Tin, Tout>::jac_adjoint_times);
}


PYBIND11_MODULE(resolvelib, m) {
    py::class_<PolarizationMatrixExponential>(m, "PolarizationMatrixExponential")
        .def(py::init<>())
        .def("apply", &PolarizationMatrixExponential::apply)
        .def("apply_with_jac", &PolarizationMatrixExponential::apply_with_jac);

    add_linearization<py::array, py::array>(m, "Linearization_field2field");
    add_linearization<py::array, py::dict >(m, "Linearization_field2mfield");
    add_linearization<py::dict , py::array>(m, "Linearization_mfield2field");
    add_linearization<py::dict , py::dict >(m, "Linearization_mfield2mfield");
}
