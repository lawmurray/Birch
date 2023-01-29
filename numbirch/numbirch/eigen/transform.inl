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
template<class T, class Functor>
void kernel_for_each(const int m, const int n, T* A, const int ldA,
    Functor f) {
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      get(A, i, j, ldA) = f(i, j);
    }
  }
}
template<class Functor>
auto for_each(const int n, Functor f) {
  auto x = Array<decltype(f(0,0)),1>(make_shape(n));
  kernel_for_each(1, n, sliced(x), stride(x), f);
  return x;
}
template<class Functor>
auto for_each(const int m, const int n, Functor f) {
  auto A = Array<decltype(f(0,0)),2>(make_shape(m, n));
  kernel_for_each(m, n, sliced(A), stride(A), f);
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
    kernel_transform(m, n, sliced(x), stride(x), sliced(y), stride(y), f);
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
    kernel_transform(m, n, sliced(x), stride(x), sliced(y), stride(y), sliced(z),
        stride(z), f);
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
    kernel_transform(m, n, sliced(x), stride(x), sliced(y), stride(y), sliced(z),
        stride(z), sliced(a), stride(a), f);
    return a;
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
      get(C, i, j, ldC) = get(A, get(I, i, j, ldI) - 1, 0, ldA);
    }
  }
}
template<class T, class U>
auto gather(const T& x, const U& i) {
  constexpr int D = dimension_v<U>;
  auto m = width(i);
  auto n = height(i);
  auto z = Array<value_t<T>,D>(make_shape<D>(m, n));
  kernel_gather(m, n, sliced(x), stride(x), sliced(i), stride(i), sliced(z),
      stride(z));
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
      get(D, i, j, ldD) = get(A, get(I, i, j, ldI) - 1, get(J, i, j, ldJ) - 1,
          ldA);
    }
  }
}
template<class T, class U, class V>
auto gather(const T& x, const U& i, const V& j) {
  constexpr int D = dimension_v<implicit_t<U,V>>;
  auto m = width(i, j);
  auto n = height(i, j);
  auto z = Array<value_t<T>,D>(make_shape<D>(m, n));
  kernel_gather(m, n, sliced(x), stride(x), sliced(i), stride(i), sliced(j),
      stride(j), sliced(z), stride(z));
  return z;
}

}
