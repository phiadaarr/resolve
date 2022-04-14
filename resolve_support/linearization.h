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
  LinearizationWithMetric(const Tout &position_, function<Tout(const Tin &)> jac_times_,
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
      .def("jac_adjoint_times", &LinearizationWithMetric<Tin>::jac_adjoint_times)
      .def("apply_metric", &LinearizationWithMetric<Tin>::apply_metric);
}
