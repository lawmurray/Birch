/**
 * @file
 */
#pragma once

#include "numbirch/array.hpp"
#include "numbirch/type.hpp"

namespace numbirch {
/*
 * Prefetch an array onto device.
 */
template<class T, int D>
void prefetch(const Array<T,D>& x) {
  //
}

/*
 * Prefetch a scalar onto device---null operation.
 */
template<class T, class = std::enable_if_t<is_arithmetic<T>::value,int>>
void prefetch(const T& x) {
  //
}

/*
 * Unary transform.
 */
template<class T, class R, class Functor>
void kernel_transform(const int m, const int n, const T A, const int ldA, R B,
    const int ldB, Functor f) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      element(B, i, j, ldB) = f(element(A, i, j, ldA));
    }
  }
}
template<class T, class Functor>
auto transform(const T& x, Functor f) {
  using R = decltype(f(value_t<T>()));
  constexpr int D = dimension_v<T>;
  auto y = Array<R,D>(shape(x));
  auto m = rows(x);
  auto n = columns(x);
  kernel_transform(m, n, data(x), stride(x), data(y), stride(y), f);
  return y;
}

/*
 * Gradient of unary transform.
 */
template<class G, class T, class Functor>
auto transform_grad(const G& g, const T& x, Functor f) {
  return transform(g, x, f);  // same as binary transform
}

/*
 * Binary transform.
 */
template<class T, class U, class R, class Functor>
void kernel_transform(const int m, const int n, const T A, const int ldA,
    const U B, const int ldB, R C, const int ldC, Functor f) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      element(C, i, j, ldC) = f(element(A, i, j, ldA), element(B, i, j, ldB));
    }
  }
}
template<class T, class U, class Functor>
auto transform(const T& x, const U& y, Functor f) {
  assert(conforms(x, y));
  using R = decltype(f(value_t<T>(),value_t<U>()));
  constexpr int D = std::max(dimension_v<T>, dimension_v<U>);
  auto m = std::max(rows(x), rows(y));
  auto n = std::max(columns(x), columns(y));
  auto z = Array<R,D>(m, n);
  kernel_transform(m, n, data(x), stride(x), data(y), stride(y), data(z),
      stride(z), f);
  return z;
}

/*
 * Gradient of binary transform.
 */
template<class G, class T, class U, class V, class W, class Functor>
void kernel_transform_grad(const int m, const int n, const G g, const int ldg,
    const T A, const int ldA, const U B, const int ldB, V GA, const int ldGA,
    W GB, const int ldGB, Functor f) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      auto pair = f(element(g, i, j, ldg), element(A, i, j, ldA),
          element(B, i, j, ldB));
      element(GA, i, j, ldGA) = pair.first;
      element(GB, i, j, ldGB) = pair.second;
    }
  }
}
template<class G, class T, class U, class Functor>
auto transform_grad(const G& g, const T& x, const U& y, Functor f) {
  assert(conforms(x, y));
  using P = decltype(f(value_t<G>(),value_t<T>(),value_t<U>()));
  using V = typename P::first_type;
  using W = typename P::second_type;
  constexpr int D = std::max(std::max(dimension_v<G>, dimension_v<T>),
      dimension_v<U>);
  auto m = std::max(std::max(rows(g), rows(x)), rows(y));
  auto n = std::max(std::max(columns(g), columns(x)), columns(y));
  auto a = Array<V,D>(m, n);
  auto b = Array<W,D>(m, n);
  kernel_transform_grad(m, n, data(g), stride(g), data(x), stride(x), data(y),
      stride(y), data(a), stride(a), data(b), stride(b), f);
  return std::make_pair(a, b);
}

/*
 * Ternary transform.
 */
template<class T, class U, class V, class R, class Functor>
void kernel_transform(const int m, const int n, const T A, const int ldA,
    const U B, const int ldB, const V C, const int ldC, R D, const int ldD,
    Functor f) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      element(D, i, j, ldD) = f(element(A, i, j, ldA), element(B, i, j, ldB),
          element(C, i, j, ldC));
    }
  }
}
template<class T, class U, class V, class Functor>
auto transform(const T& x, const U& y, const V& z, Functor f) {
  assert(conforms(x, y) && conforms(y, z));
  using R = decltype(f(value_t<T>(),value_t<U>(),value_t<V>()));
  constexpr int D = std::max(std::max(dimension_v<T>, dimension_v<U>),
      dimension_v<V>);
  auto m = std::max(std::max(rows(x), rows(y)), rows(z));
  auto n = std::max(std::max(columns(x), columns(y)), columns(z));
  auto a = Array<R,D>(m, n);
  kernel_transform(m, n, data(x), stride(x), data(y), stride(y), data(z),
      stride(z), data(a), stride(a), f);
  return a;
}

}
