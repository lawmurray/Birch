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
    assert(0 <= j - 1);
    return get(A, 0, j - 1, ldA);
  }
  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    assert(0 <= i - 1);
    assert(0 <= j - 1);
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
    int k = get(I, i, j, ldI) - 1;
    int l = get(J, i, j, ldJ) - 1;
    assert(0 <= k);
    assert(0 <= l);
    auto& c = get(C, k, l, ldC);
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
Array<real,0> fill_grad(const Array<real,1>& g, const Array<value_t<T>,1>& y,
    const T& x, const int n) {
  return sum(g);
}

template<class T, class>
Array<value_t<T>,2> fill(const T& x, const int m, const int n) {
  Array<value_t<T>,2> C(make_shape(m, n));
  for_each(m, n, fill_functor(sliced(x), sliced(C), stride(C)));
  return C;
}

template<class T, class>
Array<real,0> fill_grad(const Array<real,2>& g, const Array<value_t<T>,2>& C,
    const T& x, const int m, const int n) {
  return sum(g);
}

template<class T, class>
Array<value_t<T>,1> iota(const T& x, const int n) {
  Array<value_t<T>,1> y(make_shape(n));
  for_each(n, iota_functor(sliced(x), sliced(y), stride(y)));
  return y;
}

template<class T, class>
Array<real,0> iota_grad(const Array<real,1>& g, const Array<value_t<T>,1>& y,
    const T& x, const int n) {
  return sum(g);
}

template<class T, class>
Array<value_t<T>,2> diagonal(const T& x, const int n) {
  Array<value_t<T>,2> A(make_shape(n, n));
  for_each(n, n, diagonal_functor(sliced(x), sliced(A), stride(A)));
  return A;
}

template<class T, class>
Array<real,0> diagonal_grad(const Array<real,2>& g,
    const Array<value_t<T>,2>& y, const T& x, const int n) {
  return sum(g.diagonal());
}

template<class T, class>
Array<T,2> diagonal(const Array<T,1>& x) {
  Array<T,2> y(make_shape(length(x), length(x)), T(0));
  y.diagonal() = x;
  return y;
}

template<class T, class>
Array<real,1> diagonal_grad(const Array<real,2>& g, const Array<T,2>& y,
    const Array<T,1>& x) {
  return g.diagonal();
}

template<class T, class U, class>
Array<T,0> element(const Array<T,1>& x, const U& i) {
  return transform(i, gather_functor(sliced(x), stride(x)));
}

template<class T, class U, class>
Array<real,1> element_grad1(const Array<real,0>& g, const Array<T,0>& y,
    const Array<T,1>& x, const U& i) {
  return single(g, i, length(x));
}

template<class T, class U, class>
real element_grad2(const Array<real,0>& g, const Array<T,0>& y,
    const Array<T,1>& x, const U& i) {
  return real(0);
}

template<class T, class U, class V, class>
Array<T,0> element(const Array<T,2>& A, const U& i, const V& j) {
  return transform(i, j, gather_functor(sliced(A), stride(A)));
}

template<class T, class U, class V, class>
Array<real,2> element_grad1(const Array<real,0>& g,
    const Array<T,0>& y, const Array<T,2>& A, const U& i, const V& j) {
  return single(g, i, j, rows(A), columns(A));
}

template<class T, class U, class V, class>
real element_grad2(const Array<real,0>& g, const Array<T,0>& y,
    const Array<T,2>& A, const U& i, const V& j) {
  return real(0);
}

template<class T, class U, class V, class>
real element_grad3(const Array<real,0>& g, const Array<T,0>& y,
    const Array<T,2>& A, const U& i, const V& j) {
  return real(0);
}

template<class T, class U, class>
Array<value_t<T>,1> single(const T& x, const U& i, const int n) {
  Array<value_t<T>,1> y(make_shape(n));
  for_each(n, single_functor(sliced(x), 1, sliced(i), sliced(y), stride(y)));
  return y;
}

template<class T, class U, class>
Array<real,0> single_grad1(const Array<real,1>& g,
    const Array<value_t<T>,1>& y, const T& x, const U& i, const int n) {
  return element(g, i);
}

template<class T, class U, class>
real single_grad2(const Array<real,1>& g, const Array<value_t<T>,1>& y,
    const T& x, const U& i, const int n) {
  return real(0);
}

template<class T, class U, class V, class>
Array<value_t<T>,2> single(const T& x, const U& i, const V& j, const int m,
    const int n) {
  Array<value_t<T>,2> Y(make_shape(m, n));
  for_each(m, n, single_functor(sliced(x), sliced(i), sliced(j), sliced(Y),
      stride(Y)));
  return Y;
}

template<class T, class U, class V, class>
Array<real,0> single_grad1(const Array<real,2>& g,
    const Array<value_t<T>,2>& A, const T& x, const U& i, const V& j,
    const int m, const int n) {
  return element(g, i, j);
}

template<class T, class U, class V, class>
real single_grad2(const Array<real,2>& g, const Array<value_t<T>,2>& A,
    const T& x, const U& i, const V& j, const int m, const int n) {
  return real(0);
}

template<class T, class U, class V, class>
real single_grad3(const Array<real,2>& g, const Array<value_t<T>,2>& A,
    const T& x, const U& i, const V& j, const int m, const int n) {
  return real(0);
}

template<class T, class U, class>
pack_t<T,U> pack(const T& x, const U& y) {
  assert(rows(x) == rows(y));
  [[maybe_unused]] auto r = rows(x);
  [[maybe_unused]] auto cx = columns(x);
  [[maybe_unused]] auto cy = columns(y);
  pack_t<T,U> z(make_shape(r, cx + cy));

  if constexpr (is_scalar_v<T>) {
    z.slice(1, 1) = x;
    if constexpr (is_scalar_v<U>) {
      z.slice(1, 2) = y;
    } else if constexpr (is_vector_v<U>) {
      z.slice(1, std::make_pair(2, 2)) = y;
    } else {
      static_assert(is_matrix_v<U>);
      z.slice(std::make_pair(1, 1), std::make_pair(2, 1 + cy)) = y;
    }
  } else if constexpr (is_vector_v<T>) {
    z.slice(std::make_pair(1, r), 1) = x;
    if constexpr (is_scalar_v<U>) {
      z.slice(1, 2) = y;
    } else if constexpr (is_vector_v<U>) {
      z.slice(std::make_pair(1, r), 2) = y;
    } else {
      static_assert(is_matrix_v<U>);
      z.slice(std::make_pair(1, r), std::make_pair(2, 1 + cy)) = y;
    }
  } else {
    static_assert(is_matrix_v<T>);
    z.slice(std::make_pair(1, r), std::make_pair(1, cx)) = x;
    if constexpr (is_scalar_v<U>) {
      z.slice(1, cx + 1) = y;
    } else if constexpr (is_vector_v<U>) {
      z.slice(std::make_pair(1, r), cx + 1) = y;
    } else {
      static_assert(is_matrix_v<U>);
      z.slice(std::make_pair(1, r), std::make_pair(cx + 1, cx + cy)) = y;
    }
  }
  return z;
}

template<class T, class U, class>
real_t<T> pack_grad1(const real_t<pack_t<T,U>>& g, const pack_t<T,U>& z,
    const T& x, const U& y) {
  assert(rows(x) == rows(y));
  [[maybe_unused]] auto r = rows(x);
  [[maybe_unused]] auto cx = columns(x);

  if constexpr (is_scalar_v<T>) {
    return g.slice(1, 1);
  } else if constexpr (is_vector_v<T>) {
    return g.slice(std::make_pair(1, r), 1);
  } else {
    static_assert(is_matrix_v<T>);
    return g.slice(std::make_pair(1, r), std::make_pair(1, cx));
  }
}

template<class T, class U, class>
real_t<U> pack_grad2(const real_t<pack_t<T,U>>& g, const pack_t<T,U>& z,
    const T& x, const U& y) {
  assert(rows(x) == rows(y));
  [[maybe_unused]] auto r = rows(x);
  [[maybe_unused]] auto cx = columns(x);
  [[maybe_unused]] auto cy = columns(y);

  if constexpr (is_scalar_v<U>) {
    return g.slice(1, cx + 1);
  } else if constexpr (is_vector_v<U>) {
    return g.slice(std::make_pair(1, r), cx + 1);
  } else {
    static_assert(is_matrix_v<U>);
    return g.slice(std::make_pair(1, r), std::make_pair(cx + 1, cx + cy));
  }
}

template<class T, class U, class>
stack_t<T,U> stack(const T& x, const U& y) {
  assert(columns(x) == columns(y));
  [[maybe_unused]] auto rx = rows(x);
  [[maybe_unused]] auto ry = rows(y);
  [[maybe_unused]] auto c = columns(x);

  if constexpr (is_scalar_v<T>) {
    if constexpr (is_scalar_v<U>) {
      stack_t<T,U> z(make_shape(2));
      z.slice(1) = x;
      z.slice(2) = y;
      return z;
    } else if constexpr (is_vector_v<U>) {
      stack_t<T,U> z(make_shape(1 + ry));
      z.slice(1) = x;
      z.slice(std::make_pair(2, 1 + ry)) = y;
      return z;
    } else {
      static_assert(is_matrix_v<U>);
      stack_t<T,U> z(make_shape(1 + ry, 1));
      z.slice(1, 1) = x;
      z.slice(std::make_pair(2, 1 + ry), std::make_pair(1, 1)) = y;
      return z;
    }
  } else if constexpr (is_vector_v<T>) {
    if constexpr (is_scalar_v<U>) {
      stack_t<T,U> z(make_shape(rx + 1));
      z.slice(std::make_pair(1, rx)) = x;
      z.slice(rx + 1) = y;
      return z;
    } else if constexpr (is_vector_v<U>) {
      stack_t<T,U> z(make_shape(rx + ry));
      z.slice(std::make_pair(1, rx)) = x;
      z.slice(std::make_pair(rx + 1, rx + ry)) = y;
      return z;
    } else {
      static_assert(is_matrix_v<U>);
      stack_t<T,U> z(make_shape(rx + ry, 1));
      z.slice(std::make_pair(1, rx), 1) = x;
      z.slice(std::make_pair(rx + 1, rx + ry), std::make_pair(1, 1)) = y;
      return z;
    }
  } else {
    static_assert(is_matrix_v<T>);
    stack_t<T,U> z(make_shape(rx + ry, c));
    z.slice(std::make_pair(1, rx), std::make_pair(1, c)) = x;
    if constexpr (is_scalar_v<U>) {
      z.slice(rx + 1, 1) = y;
    } else if constexpr (is_vector_v<U>) {
      z.slice(std::make_pair(rx + 1, rx + ry), 1) = y;
    } else {
      static_assert(is_matrix_v<U>);
      z.slice(std::make_pair(rx + 1, rx + ry), std::make_pair(1, c)) = y;
    }
    return z;
  }
}

template<class T, class U, class>
real_t<T> stack_grad1(const real_t<stack_t<T,U>>& g, const stack_t<T,U>& z,
    const T& x, const U& y) {
  assert(columns(x) == columns(y));
  [[maybe_unused]] auto rx = rows(x);
  [[maybe_unused]] auto c = columns(x);

  if constexpr (is_scalar_v<T>) {
    if constexpr (is_matrix_v<stack_t<T,U>>) {
      return g.slice(1, 1);
    } else {
      return g.slice(1);
    }
  } else if constexpr (is_vector_v<T>) {
    if constexpr (is_matrix_v<stack_t<T,U>>) {
      return g.slice(std::make_pair(1, rx), 1);
    } else {
      return g.slice(std::make_pair(1, rx));
    }
  } else {
    static_assert(is_matrix_v<stack_t<T,U>>);
    return g.slice(std::make_pair(1, rx), std::make_pair(1, c));
  }
}

template<class T, class U, class>
real_t<U> stack_grad2(const real_t<stack_t<T,U>>& g, const stack_t<T,U>& z,
    const T& x, const U& y) {
  assert(columns(x) == columns(y));
  [[maybe_unused]] auto rx = rows(x);
  [[maybe_unused]] auto ry = rows(y);
  [[maybe_unused]] auto c = columns(x);

  if constexpr (is_scalar_v<U>) {
    if constexpr (is_matrix_v<stack_t<T,U>>) {
      return g.slice(rx + 1, 1);
    } else {
      return g.slice(rx + 1);
    }
  } else if constexpr (is_vector_v<U>) {
    if constexpr (is_matrix_v<stack_t<T,U>>) {
      return g.slice(std::make_pair(rx + 1, rx + ry), 1);
    } else {
      return g.slice(std::make_pair(rx + 1, rx + ry));
    }
  } else {
    static_assert(is_matrix_v<stack_t<T,U>>);
    return g.slice(std::make_pair(rx + 1, rx + ry), std::make_pair(1, c));
  }
}

template<class T, class>
Array<value_t<T>,1> vec(const T& x) {
  if constexpr (is_vector_v<T>) {
    return x;
  } else if constexpr (is_arithmetic_v<T>) {
    return Array<value_t<T>,1>(x);
  } else if (x.canReshape()) {
    return Array<value_t<T>,1>(x.control(), make_shape(x.size()), false);
  } else {
    Array<value_t<T>,1> y(make_shape(size(x)));
    for_each(size(x), reshape_functor(width(x), 1, sliced(x), stride(x),
        sliced(y), stride(y)));
    return y;
  }
}

template<class T, class>
real_t<T> vec_grad(const Array<real,1>& g, const Array<value_t<T>,1>& y,
    const T& x) {
  if constexpr (is_scalar_v<T>) {
    return g.slice(1);
  } else if constexpr (is_vector_v<T>) {
    return g;
  } else {
    return mat(g, columns(x));
  }
}

template<class T, class>
Array<value_t<T>,2> mat(const T& x, const int n) {
  assert(size(x) % n == 0);
  if constexpr (is_matrix_v<T>) {
    return x;
  } else if constexpr (is_arithmetic_v<T>) {
    return Array<value_t<T>,2>(x);
  } else if (x.canReshape()) {
    return Array<value_t<T>,2>(x.control(), make_shape(size(x)/n, n), false);
  } else {
    Array<value_t<T>,2> y(make_shape(size(x)/n, n));
    for_each(size(x)/n, n, reshape_functor(width(x), size(x)/n, sliced(x),
        stride(x), sliced(y), stride(y)));
    return y;
  }
}

template<class T, class>
real_t<T> mat_grad(const Array<real,2>& g, const Array<value_t<T>,2>& y,
    const T& x, const int n) {
  if constexpr (is_scalar_v<T>) {
    return g.slice(1, 1);
  } else if constexpr (is_vector_v<T>) {
    return vec(g);
  } else {
    return mat(g, columns(x));
  }
}

template<class T, class>
Array<T,1> gather(const Array<T,1>& x, const Array<int,1>& y) {
  return transform(y, gather_functor(sliced(x), stride(x)));
}

template<class T, class>
Array<real,1> gather_grad1(const Array<real,1>& g, const Array<T,1>& z,
    const Array<T,1>& x, const Array<int,1>& y) {
  return scatter(g, y, length(x));
}

template<class T, class>
Array<real,1> gather_grad2(const Array<real,1>& g, const Array<T,1>& z,
    const Array<T,1>& x, const Array<int,1>& y) {
  return fill(real(0), length(y));
}

template<class T, class>
Array<T,2> gather(const Array<T,2>& A, const Array<int,2>& I,
    const Array<int,2>& J) {
  return transform(I, J, gather_functor(sliced(A), stride(A)));
}

template<class T, class>
Array<real,2> gather_grad1(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J) {
  return scatter(G, I, J, rows(A), columns(A));
}

template<class T, class>
Array<real,2> gather_grad2(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J) {
  return fill(real(0), rows(I), columns(I));
}

template<class T, class>
Array<real,2> gather_grad3(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J) {
  return fill(real(0), rows(J), columns(J));
}

template<class T, class>
Array<T,1> scatter(const Array<T,1>& x, const Array<int,1>& y, const int n) {
  assert(conforms(x, y));
  auto z = fill(T(0), n);
  for_each(length(x), scatter_functor(sliced(x), stride(x), 1, 0, sliced(y),
      stride(y), sliced(z), stride(z)));
  return z;
}

template<class T, class>
Array<real,1> scatter_grad1(const Array<real,1>& g, const Array<T,1>& z,
    const Array<T,1>& x, const Array<int,1>& y, const int n) {
  return gather(g, y);
}

template<class T, class>
Array<real,1> scatter_grad2(const Array<real,1>& g, const Array<T,1>& z,
    const Array<T,1>& x, const Array<int,1>& y, const int n) {
  return Array<real,1>(0, shape(y));
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

template<class T, class>
Array<real,2> scatter_grad1(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J,
    const int m, const int n) {
  return gather(G, I, J);
}

template<class T, class>
Array<real,2> scatter_grad2(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J,
    const int m, const int n) {
  return fill(real(0), rows(I), columns(I));
}

template<class T, class>
Array<real,2> scatter_grad3(const Array<real,2>& G, const Array<T,2>& C,
    const Array<T,2>& A, const Array<int,2>& I, const Array<int,2>& J,
    const int m, const int n) {
  return fill(real(0), rows(J), columns(J));
}

}
