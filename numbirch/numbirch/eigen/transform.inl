/**
 * @file
 */
#pragma once

#include "numbirch/eigen/eigen.hpp"
#include "numbirch/array.hpp"
#include "numbirch/utility.hpp"

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
template<class Functor>
void kernel_for_each(const int m, const int n, Functor f) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      f(i, j);
    }
  }
}
template<class Functor>
void for_each(const int n, Functor f) {
  kernel_for_each(1, n, f);
}
template<class Functor>
void for_each(const int m, const int n, Functor f) {
  kernel_for_each(m, n, f);
}

/*
 * Unary transform.
 */
template<class T, class R, class Functor>
void kernel_transform(const int m, const int n, const T A, const int ldA, R B,
    const int ldB, Functor f) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      get(B, i, j, ldB) = f(get(A, i, j, ldA));
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
    auto m = width(x);
    auto n = height(x);
    auto y = Array<R,D>(make_shape<D>(m, n));
    kernel_transform(m, n, buffer(x), stride(x), buffer(y), stride(y), f);
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
      get(C, i, j, ldC) = f(get(A, i, j, ldA), get(B, i, j, ldB));
    }
  }
}
template<class T, class U, class Functor>
auto transform(const T& x, const U& y, Functor f) {
  if constexpr (is_arithmetic_v<T> && is_arithmetic_v<U>) {
    return f(x, y);
  } else {
    using R = decltype(f(value_t<T>(),value_t<U>()));
    constexpr int D = dimension_v<implicit_t<T,U>>;
    auto m = width(x, y);
    auto n = height(x, y);
    auto z = Array<R,D>(make_shape<D>(m, n));
    kernel_transform(m, n, buffer(x), stride(x), buffer(y), stride(y),
        buffer(z), stride(z), f);
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
      get(D, i, j, ldD) = f(get(A, i, j, ldA), get(B, i, j, ldB),
          get(C, i, j, ldC));
    }
  }
}
template<class T, class U, class V, class Functor>
auto transform(const T& x, const U& y, const V& z, Functor f) {
  if constexpr (is_arithmetic_v<T> && is_arithmetic_v<U> &&
      is_arithmetic_v<V>) {
    return f(x, y, z);
  } else {
    using R = decltype(f(value_t<T>(),value_t<U>(),value_t<V>()));
    constexpr int D = dimension_v<implicit_t<T,U,V>>;
    auto m = width(x, y, z);
    auto n = height(x, y, z);
    auto a = Array<R,D>(make_shape<D>(m, n));
    kernel_transform(m, n, buffer(x), stride(x), buffer(y), stride(y), buffer(z),
        stride(z), buffer(a), stride(a), f);
    return a;
  }
}

}
