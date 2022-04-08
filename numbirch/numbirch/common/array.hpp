/**
 * @file
 */
#pragma once

#include "numbirch/array.hpp"
#include "numbirch/common/functor.hpp"

namespace numbirch {
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
