/**
 * @file
 */
#pragma once

#include "numbirch/array.hpp"

namespace numbirch {
template<class T, class U>
struct fill_functor {
  const T a;
  U A;
  const int ldA;
  fill_functor(const T a, U A, const int ldA) : a(a), A(A), ldA(ldA) {

  }
  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    get(A, i, j, ldA) = get(a);
  }
};

template<class T, class U>
struct iota_functor {
  const T a;
  U A;
  const int ldA;
  iota_functor(const T a, U A, const int ldA) : a(a), A(A), ldA(ldA) {
    //
  }
  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    get(A, i, j, ldA) = get(a) + j;
  }
};

template<class T, class U>
struct diagonal_functor {
  const T a;
  U A;
  const int ldA;
  diagonal_functor(const T a, U A, const int ldA) : a(a), A(A), ldA(ldA) {
    //
  }
  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    get(A, i, j, ldA) = (i == j) ? get(a) : 0;
  }
};

template<class T, class U, class V, class W>
struct single_functor {
  const T x;
  const U k;
  const V l;
  W A;
  const int ldA;
  single_functor(const T x, const U k, const V l, W A, const int ldA) :
      x(x), k(k), l(l), A(A), ldA(ldA) {
    //
  }
  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    get(A, i, j, ldA) = (i == get(k) - 1 && j == get(l) - 1) ? get(x) : 0;
  }
};

template<class T, class U>
struct reshape_functor {
  /**
   * Width of source array.
   */
  const int m1;

  /**
   * Width of destination array.
   */
  const int m2;

  const T A;
  const int ldA;
  const U C;
  const int ldC;

  reshape_functor(const int m1, const int m2, const T A, const int ldA, U C,
      const int ldC) : m1(m1), m2(m2), A(A), ldA(ldA), C(C), ldC(ldC) {
    //
  }
  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    // (i,j) is the element in the destination, (k,l) in the source
    auto s = i + j*m2;  // serial index
    auto k = s % m1;
    auto l = s / m1;
    get(C, i, j, ldC) = get(A, k, l, ldA);
  }
};

template<class T>
struct gather_functor {
  const T A;
  const int ldA;
  gather_functor(const T A, const int ldA) : A(A), ldA(ldA) {
    //
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int j) const {
    return get(A, 0, j - 1, ldA);
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    return get(A, i - 1, j - 1, ldA);
  }
};

template<class T, class U, class V>
struct scatter_functor {
  T A;
  const int ldA;
  U I;
  const int ldI;
  V J;
  const int ldJ;
  T C;
  const int ldC;

  scatter_functor(T A, const int ldA, const U I, const int ldI, const V J,
      const int ldJ, const T C, const int ldC) : A(A), ldA(ldA), I(I), ldI(ldI),
      J(J), ldJ(ldJ), C(C), ldC(ldC) {
    //
  }
  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    auto& c = get(C, get(I, i, j, ldI) - 1, get(J, i, j, ldJ) - 1, ldC);
    auto a = get(A, i, j, ldA);
    #ifdef __CUDA_ARCH__
    if constexpr (is_bool_v<typeof(a)>) {
      /* atomicOr() does not support bool, but as all c are initialized to
       * zero (false), we achieve logical or by writing only if true */
      if (a) {
        c = a;
      }
    } else {
      atomicAdd(&c, a);
    }
    #else
    c += a;
    #endif
  }
};

template<class T, class>
Array<value_t<T>,1> fill(const T& x, const int n) {
  Array<value_t<T>,1> y(make_shape(n));
  for_each(n, fill_functor(sliced(x), sliced(y), stride(y)));
  return y;
}

template<class T, class>
Array<value_t<T>,2> fill(const T& x, const int m, const int n) {
  Array<value_t<T>,2> C(make_shape(m, n));
  for_each(m, n, fill_functor(sliced(x), sliced(C), stride(C)));
  return C;
}

template<class T, class>
Array<value_t<T>,1> iota(const T& x, const int n) {
  Array<value_t<T>,1> y(make_shape(n));
  for_each(n, iota_functor(sliced(x), sliced(y), stride(y)));
  return y;
}

template<class T, class>
Array<value_t<T>,2> diagonal(const T& x, const int n) {
  Array<value_t<T>,2> A(make_shape(n, n));
  for_each(n, n, diagonal_functor(sliced(x), sliced(A), stride(A)));
  return A;
}

template<class T, class U, class>
Array<T,0> element(const Array<T,1>& x, const U& i) {
  return transform(i, gather_functor(sliced(x), stride(x)));
}

template<class T, class U, class V, class>
Array<T,0> element(const Array<T,2>& A, const U& i, const V& j) {
  return transform(i, j, gather_functor(sliced(A), stride(A)));
}

template<class T, class U, class>
Array<value_t<T>,1> single(const T& x, const U& i, const int n) {
  Array<value_t<T>,1> y(make_shape(n));
  for_each(n, single_functor(sliced(x), 1, sliced(i), sliced(y), stride(y)));
  return y;
}

template<class T, class U, class V, class>
Array<value_t<T>,2> single(const T& x, const U& i, const V& j, const int m,
    const int n) {
  Array<value_t<T>,2> Y(make_shape(m, n));
  for_each(m, n, single_functor(sliced(x), sliced(i), sliced(j), sliced(Y),
      stride(Y)));
  return Y;
}

template<class T, class>
Array<value_t<T>,1> vec(const T& x) {
  Array<value_t<T>,1> y(make_shape(size(x)));
  for_each(size(x), reshape_functor(width(x), 1, sliced(x), stride(x),
      sliced(y), stride(y)));
  return y;
}

template<class T, class>
Array<value_t<T>,2> mat(const T& x, const int n) {
  assert(size(x) % n == 0);
  Array<value_t<T>,2> y(make_shape(size(x)/n, n));
  for_each(size(x)/n, n, reshape_functor(width(x), size(x)/n, sliced(x),
      stride(x), sliced(y), stride(y)));
  return y;
}

template<class T, class>
Array<T,1> gather(const Array<T,1>& x, const Array<int,1>& y) {
  return transform(y, gather_functor(sliced(x), stride(x)));
}

template<class T, class>
Array<T,2> gather(const Array<T,2>& A, const Array<int,2>& I,
    const Array<int,2>& J) {
  return transform(I, J, gather_functor(sliced(A), stride(A)));
}

template<class T, class>
Array<T,1> scatter(const Array<T,1>& x, const Array<int,1>& y, const int n) {
  assert(conforms(x, y));
  auto z = fill(T(0), n);
  for_each(length(x), scatter_functor(sliced(x), stride(x), 0, 0, sliced(y),
      stride(y), sliced(z), stride(z)));
  return z;
}

template<class T, class>
Array<T,2> scatter(const Array<T,2>& A, const Array<int,2>& I,
    const Array<int,2>& J, const int m, const int n) {
  assert(conforms(A, I));
  assert(conforms(A, J));
  auto C = fill(T(0), m, n);
  for_each(rows(A), columns(A), scatter_functor(sliced(A), stride(A),
      sliced(I), stride(I), sliced(J), stride(J), sliced(C), stride(C)));
  return C;
}

}
