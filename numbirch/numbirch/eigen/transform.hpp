/**
 * @file
 */
#pragma once

#include "numbirch/eigen/eigen.hpp"
#include "numbirch/type.hpp"

namespace numbirch {

template<class T, int D>
void prefetch(const Array<T,D>& x) {
  //
}

template<class T, class = std::enable_if_t<is_arithmetic<T>::value,int>>
void prefetch(const T& x) {
  //
}

template<class T, class Functor>
auto transform(const T& x, Functor f) {
  using R = decltype(f(value_t<T>()));
  constexpr int D = dimension_v<T>;
  auto y = Array<R,D>(shape(x));
  auto x1 = make_eigen_matrix(x);
  auto y1 = make_eigen_matrix(y);
  y1.noalias() = x1.unaryExpr(f);
  return y1;
}

template<class G, class T, class Functor>
auto transform_grad(const G& g, const T& x, Functor f) {
  return transform(g, x, f);  // same as binary transform
}

template<class T, class U, class Functor>
auto transform(const T& x, const U& y, Functor f) {
  assert(conforms(x, y));
  using R = decltype(f(value_t<T>(),value_t<U>()));
  constexpr int D = std::max(dimension_v<T>, dimension_v<U>);
  auto m = std::max(rows(x), rows(y));
  auto n = std::max(columns(x), columns(y));
  auto z = Array<R,D>(m, n);

  auto x1 = make_eigen_matrix(x);
  auto y1 = make_eigen_matrix(y);
  auto z1 = make_eigen_matrix(z);
  z1.noalias() = x1.binaryExpr(y1, f);
}

template<class G, class T, class U, class Functor>
auto transform_grad(const G& g, const T& x, const U& y, Functor f) {
  assert(conforms(x, y));

}

template<class T, class U, class V, class Functor>
auto transform(const T& x, const U& y, const V& z, Functor f) {
  assert(conforms(x, y) && conforms(y, z));

}

}
