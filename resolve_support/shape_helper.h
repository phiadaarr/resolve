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

/* Copyright (C) 2020-2022 Max-Planck-Society
   Author: Martin Reinecke */

#include <algorithm>
#include <array>
#include <iostream>

using namespace std;

using shape_t = vector<size_t>;

template <typename T, size_t L1, size_t L2>
array<T, L1 + L2> combine_shapes(const array<T, L1> &a1, const array<T, L2> &a2) {
  array<T, L1 + L2> res;
  copy_n(a1.begin(), L1, res.begin());
  copy_n(a2.begin(), L2, res.begin() + L1);
  return res;
}
template <typename T, size_t L>
array<T, 1 + L> combine_shapes(size_t s1, const array<T, L> &a) {
  array<T, 1 + L> res;
  res[0] = s1;
  copy_n(a.begin(), L, res.begin() + 1);
  return res;
}
template <typename T, size_t L>
array<T, 1 + L> combine_shapes(const array<T, L> &a, size_t s2) {
  array<T, 1 + L> res;
  copy_n(a.begin(), L, res.begin());
  res[L] = s2;
  return res;
}

shape_t combine_shapes(const shape_t &vec, const size_t s2) {
  shape_t out;
  for (auto i : vec)
    out.push_back(i);
  out.push_back(s2);
  return out;
}

shape_t combine_shapes(const size_t s1, const shape_t &vec) {
  shape_t out;
  out.push_back(s1);
  for (auto i : vec)
    out.push_back(i);
  return out;
}

ostream &operator<<(ostream &os, const shape_t &shp) {
  for (auto i : shp)
    os << i << " ";
  os << endl;
  return os;
}

shape_t copy_shape(const py::array &arr) {
  shape_t res(size_t(arr.ndim()));
  for (size_t i = 0; i < res.size(); ++i)
    res[i] = size_t(arr.shape(int(i)));
  return res;
}
