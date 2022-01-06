/**
 * @file
 */
#pragma once

#include "numbirch/eigen/eigen.hpp"
#include "numbirch/array.hpp"
#include "numbirch/type.hpp"
#include "numbirch/common/element.hpp"

namespace numbirch {
template<class T, int D>
void prefetch(const Array<T,D>& x) {
  //
}

template<class T, class = std::enable_if_t<is_arithmetic<T>::value,int>>
void prefetch(const T& x) {
  //
}

/*
 * For-each.
 */
template<class T, class Functor>
void kernel_for_each(const int m, const int n, T* A, const int ldA,
    Functor f) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      element(A, i, j, ldA) = f(i, j);
    }
  }
}
template<class Functor>
auto for_each(const int n, Functor f) {
  auto x = Array<decltype(f(0,0)),1>(make_shape(n));
  kernel_for_each(1, n, data(x), stride(x), f);
  return x;
}
template<class Functor>
auto for_each(const int m, const int n, Functor f) {
  auto A = Array<decltype(f(0,0)),2>(make_shape(m, n));
  kernel_for_each(m, n, data(A), stride(A), f);
  return A;
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
  if constexpr (is_arithmetic_v<T>) {
    return f(x);
  } else {
    using R = decltype(f(value_t<T>()));
    constexpr int D = dimension_v<T>;
    auto y = Array<R,D>(shape(x));
    kernel_transform(width(x), height(x), data(x), stride(x), data(y),
        stride(y), f);
    return y;
  }
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
  static_assert(dimension_v<T> == dimension_v<U>);
  assert(width(x) == width(y));
  assert(height(x) == height(y));

  if constexpr (is_arithmetic_v<T> && is_arithmetic_v<U>) {
    return f(x, y);
  } else {
    using R = decltype(f(value_t<T>(),value_t<U>()));
    constexpr int D = dimension_v<T>;
    auto z = Array<R,D>(shape(x));
    kernel_transform(width(x), height(x), data(x), stride(x), data(y),
        stride(y), data(z), stride(z), f);
    return z;
  }
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
  static_assert(dimension_v<T> == dimension_v<U>);
  static_assert(dimension_v<T> == dimension_v<V>);
  assert(width(x) == width(y));
  assert(width(x) == width(z));
  assert(height(x) == height(y));
  assert(height(x) == height(z));

  if constexpr (is_arithmetic_v<T> && is_arithmetic_v<U> &&
      is_arithmetic_v<V>) {
    return f(x, y, z);
  } else {
    using R = decltype(f(value_t<T>(),value_t<U>(),value_t<V>()));
    constexpr int D = dimension_v<T>;
    auto a = Array<R,D>(shape(x));
    kernel_transform(width(x), height(x), data(x), stride(x), data(y),
        stride(y), data(z), stride(z), data(a), stride(a), f);
    return a;
  }
}

/*
 * Ternary transform returning pair.
 */
template<class G, class T, class U, class V, class W, class Functor>
void kernel_transform_pair(const int m, const int n, const G g, const int ldg,
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
auto transform_pair(const G& g, const T& x, const U& y, Functor f) {
  static_assert(dimension_v<G> == dimension_v<T>);
  static_assert(dimension_v<G> == dimension_v<U>);
  assert(width(g) == width(x));
  assert(width(g) == width(y));
  assert(height(g) == height(x));
  assert(height(g) == height(y));

  if constexpr (is_arithmetic_v<G> && is_arithmetic_v<T> &&
      is_arithmetic_v<U>) {
    auto [a, b] = f(g, x, y);
    return std::make_pair(a, b);
  } else {
    constexpr int D = dimension_v<G>;
    auto a = Array<real,D>(shape(x));
    auto b = Array<real,D>(shape(y));
    kernel_transform_pair(width(x), height(x), data(g), stride(g), data(x),
        stride(x), data(y), stride(y), data(a), stride(a), data(b), stride(b),
        f);
    return std::make_pair(a, b);
  }
}

/*
 * Unary gather.
 */
template<class T, class U, class R>
void kernel_gather(const int m, const int n, const T A, const int ldA,
    const U I, const int ldI, R C, const int ldC) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      element(C, i, j, ldC) = element(A, element(I, i, j, ldI) - 1, 0, ldA);
    }
  }
}
template<class T, class U>
auto gather(const T& x, const U& i) {
  constexpr int D = dimension_v<U>;
  auto z = Array<value_t<T>,D>(shape(i));
  kernel_gather(width(i), height(i), data(x), stride(x), data(i), stride(i),
      data(z), stride(z));
  return z;
}

/*
 * Binary gather.
 */
template<class T, class U, class V, class R>
void kernel_gather(const int m, const int n, const T A, const int ldA,
    const U I, const int ldI, const V J, const int ldJ, R D, const int ldD) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      element(D, i, j, ldD) = element(A, element(I, i, j, ldI) - 1,
          element(J, i, j, ldJ) - 1, ldA);
    }
  }
}
template<class T, class U, class V>
auto gather(const T& x, const U& i, const V& j) {
  static_assert(dimension_v<U> == dimension_v<V>);
  assert(width(i) == width(j));
  assert(height(i) == height(j));

  constexpr int D = dimension_v<U>;
  auto z = Array<value_t<T>,D>(shape(i));
  kernel_gather(width(i), height(i), data(x), stride(x), data(i), stride(i),
      data(j), stride(j), data(z), stride(z));
  return z;
}

}
