// Author: Martin Reinecke

#include <array>
#include <algorithm>

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
