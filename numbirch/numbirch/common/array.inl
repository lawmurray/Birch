/**
 * @file
 */
#pragma once

#include "numbirch/array.hpp"

namespace numbirch {
template<class T>
struct diagonal_functor {
  const T a;
  diagonal_functor(const T a) :
      a(a) {
    //
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    return (i == j) ? get(a) : 0;
  }
};

template<class T, class U, class V = int>
struct single_functor {
  const T x;
  const U k;
  const V l;
  single_functor(const T& x, const U& k, const V& l) :
      x(x), k(k), l(l) {
    //
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    return (i == get(k) - 1 && j == get(l) - 1) ? get(x) : 0;
  }
};

template<class T, class>
Array<value_t<T>,2> diagonal(const T& x, const int n) {
  return for_each(n, n, diagonal_functor(data(x)));
}

template<class T, class U, class>
Array<T,0> element(const Array<T,1>& x, const U& i) {
  return gather(x, i);
}

template<class T, class U, class V, class>
Array<T,0> element(const Array<T,2>& A, const U& i, const V& j) {
  return gather(A, i, j);
}

template<class T, class U, class>
Array<value_t<T>,1> single(const T& x, const U& i, const int n) {
  return for_each(n, single_functor(data(x), 1, data(i)));
}

template<class T, class U, class V, class>
Array<value_t<T>,2> single(const T& x, const U& i, const V& j, const int m,
    const int n) {
  return for_each(m, n, single_functor(data(x), data(i), data(j)));
}

}
