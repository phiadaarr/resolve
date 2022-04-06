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

/* Copyright (C) 2022 Max-Planck-Society
   Author: Martin Reinecke */

#include <array>
#include <algorithm>
#include <iostream>

using namespace std;


template<typename T, size_t L1, size_t L2>
  array<T,L1+L2> combine_shapes(const array<T, L1> &a1, const array<T, L2> &a2)
  {
  array<T, L1+L2> res;
  copy_n(a1.begin(), L1, res.begin());
  copy_n(a2.begin(), L2, res.begin()+L1);
  return res;
  }
template<typename T, size_t L>
  array<T,1+L> combine_shapes(size_t s1, const array<T, L> &a)
  {
  array<T, 1+L> res;
  res[0] = s1;
  copy_n(a.begin(), L, res.begin()+1);
  return res;
  }
template<typename T, size_t L>
  array<T,1+L> combine_shapes(const array<T, L> &a, size_t s2)
  {
  array<T, 1+L> res;
  copy_n(a.begin(), L, res.begin());
  res[L] = s2;
  return res;
  }

vector<size_t> combine_shapes(const vector<size_t> &vec, const size_t s2)
{
vector<size_t> out;
for (auto i: vec)
  out.push_back(i);
out.push_back(s2);
return out;
}

vector<size_t> combine_shapes(const size_t s1, const vector<size_t> &vec)
{
vector<size_t> out;
out.push_back(s1);
for (auto i: vec)
  out.push_back(i);
return out;
}

ostream& operator<<(ostream& os, const vector<size_t> &shp)
{
  for (auto i: shp)
    cout << i << " ";
  cout << endl;
  return os;
}
