/**
 * @file
 */
#pragma once

#include "numbirch/array.hpp"

namespace numbirch {
template<class T>
struct fill_functor {
  const T a;
  fill_functor(const T a) :
      a(a) {
    //
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    return get(a);
  }
};

template<class T>
struct iota_functor {
  const T a;
  iota_functor(const T a) :
      a(a) {
    //
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    return get(a) + j;
  }
};

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

template<class T>
struct reshape_functor {
  /**
   * Width of source array.
   */
  const int m1;

  /**
   * Width of destination array.
   */
  const int m2;

  /**
   * Source buffer.
   */
  const T A;

  /**
   * Source stride.
   */
  const int ldA;

  reshape_functor(const int m1, const int m2, const T& A, const int ldA) :
      m1(m1), m2(m2), A(A), ldA(ldA) {
    //
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    // (i,j) is the element in the destination, (k,l) in the source
    auto s = i + j*m2;  // serial index
    auto k = s % m1;
    auto l = s / m1;
    return get(A, k, l, ldA);
  }
};

template<class T, class>
Array<value_t<T>,1> fill(const T& x, const int n) {
  return for_each(n, fill_functor(sliced(x)));
}

template<class T, class>
Array<value_t<T>,1> iota(const T& x, const int n) {
  return for_each(n, iota_functor(sliced(x)));
}

template<class T, class>
Array<value_t<T>,2> diagonal(const T& x, const int n) {
  return for_each(n, n, diagonal_functor(sliced(x)));
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
  return for_each(n, single_functor(sliced(x), 1, sliced(i)));
}

template<class T, class U, class V, class>
Array<value_t<T>,2> single(const T& x, const U& i, const V& j, const int m,
    const int n) {
  return for_each(m, n, single_functor(sliced(x), sliced(i), sliced(j)));
}

template<class T, class>
Array<value_t<T>,1> vec(const T& x) {
  return for_each(size(x), reshape_functor(width(x), 1, sliced(x), stride(x)));
}

template<class T, class>
Array<value_t<T>,2> mat(const T& x, const int n) {
  assert(size(x) % n == 0);
  return for_each(size(x)/n, n, reshape_functor(width(x), size(x)/n, sliced(x),
      stride(x)));
}

}
