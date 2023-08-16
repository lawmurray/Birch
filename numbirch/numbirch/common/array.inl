/**
 * @file
 */
#pragma once

#include "numbirch/array.hpp"

/* redundant template deduction guides are provided for functors here as a
 * workaround for missing support for P1816 (class template argument deduction
 * for aggregate types) in clang++ as of 7 May 2023; they may be removable in
 * future */

namespace numbirch {
template<class T, class U>
struct fill_functor {
  const T a;
  U A;
  const int ldA;

  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    get(A, i, j, ldA) = get(a);
  }
};
template<class T, class U>
fill_functor(T, U, int) -> fill_functor<T, U>;

template<class T, class U>
struct iota_functor {
  const T a;
  U A;
  const int ldA;

  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    get(A, i, j, ldA) = get(a) + j;
  }
};
template<class T, class U>
iota_functor(T, U, int) -> iota_functor<T, U>;

template<class T, class U>
struct diagonal_functor {
  const T a;
  U A;
  const int ldA;

  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    get(A, i, j, ldA) = (i == j) ? get(a) : 0;
  }
};
template<class T, class U>
diagonal_functor(T, U, int) -> diagonal_functor<T, U>;

template<class T, class U, class V, class W>
struct single_functor {
  const T x;
  const U k;
  const V l;
  W A;
  const int ldA;

  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    get(A, i, j, ldA) = (i == get(k) - 1 && j == get(l) - 1) ? get(x) : 0;
  }
};
template<class T, class U, class V, class W>
single_functor(T, U, V, W, int) -> single_functor<T, U, V, W>;

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

  NUMBIRCH_HOST_DEVICE void operator()(const int i, const int j) const {
    // (i,j) is the element in the destination, (k,l) in the source
    auto s = i + j*m2;  // serial index
    auto k = s % m1;
    auto l = s / m1;
    get(C, i, j, ldC) = get(A, k, l, ldA);
  }
};
template<class T, class U>
reshape_functor(int, int, T, int, U, int) -> reshape_functor<T, U>;

template<class T, class U, class V, class W>
struct gather_functor {
  const T A;
  const int ldA;
  const U I;
  const int ldI;
  const V J;
  const int ldJ;
  W C;
  const int ldC;

  NUMBIRCH_HOST_DEVICE auto operator()(const int i, const int j) const {
    get(C, i, j, ldC) = get(A, get(I, i, j, ldI) - 1, get(J, i, j, ldJ) - 1, ldA);
  }
};
template<class T, class U, class V, class W>
gather_functor(T, int, U, int, V, int, W, int) -> gather_functor<T, U, V, W>;

template<class T, class U, class V, class W>
struct scatter_functor {
  const T A;
  const int ldA;
  const U I;
  const int ldI;
  const V J;
  const int ldJ;
  W C;
  const int ldC;

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
template<class T, class U, class V, class W>
scatter_functor(T, int, U, int, V, int, W, int) -> scatter_functor<T, U, V, W>;

template<scalar T>
NUMBIRCH_KEEP Array<value_t<T>,1> fill(const T& x, const int n) {
  Array<value_t<T>,1> y(make_shape(n));
  for_each(n, fill_functor{buffer(x), buffer(y), stride(y)});
  return y;
}

template<scalar T>
NUMBIRCH_KEEP Array<real,0> fill_grad(const Array<real,1>& g, const T& x, const int n) {
  return sum(g);
}

template<scalar T>
NUMBIRCH_KEEP Array<value_t<T>,2> fill(const T& x, const int m, const int n) {
  Array<value_t<T>,2> C(make_shape(m, n));
  for_each(m, n, fill_functor{buffer(x), buffer(C), stride(C)});
  return C;
}

template<scalar T>
NUMBIRCH_KEEP Array<real,0> fill_grad(const Array<real,2>& g, const T& x, const int m,
    const int n) {
  return sum(g);
}

template<scalar T>
NUMBIRCH_KEEP Array<value_t<T>,1> iota(const T& x, const int n) {
  Array<value_t<T>,1> y(make_shape(n));
  for_each(n, iota_functor{buffer(x), buffer(y), stride(y)});
  return y;
}

template<scalar T>
NUMBIRCH_KEEP Array<real,0> iota_grad(const Array<real,1>& g, const T& x, const int n) {
  return sum(g);
}

template<scalar T>
NUMBIRCH_KEEP Array<value_t<T>,2> diagonal(const T& x, const int n) {
  Array<value_t<T>,2> A(make_shape(n, n));
  for_each(n, n, diagonal_functor{buffer(x), buffer(A), stride(A)});
  return A;
}

template<scalar T>
NUMBIRCH_KEEP Array<real,0> diagonal_grad(const Array<real,2>& g, const T& x, const int n) {
  return sum(g.diagonal());
}

template<arithmetic T>
NUMBIRCH_KEEP Array<T,2> diagonal(const Array<T,1>& x) {
  Array<T,2> y(T(0), make_shape(length(x), length(x)));
  y.diagonal() = x;
  return y;
}

template<arithmetic T>
NUMBIRCH_KEEP Array<real,1> diagonal_grad(const Array<real,2>& g, const Array<T,1>& x) {
  return g.diagonal();
}

template<arithmetic T, scalar U>
NUMBIRCH_KEEP Array<T,0> element(const Array<T,1>& x, const U& i) {
  Array<T,0> z;
  for_each(1, 1, gather_functor{buffer(x), stride(x), 1, 0, buffer(i),
      stride(i), buffer(z), stride(z)});
  return z;
}

template<arithmetic T, scalar U>
NUMBIRCH_KEEP Array<real,1> element_grad1(const Array<real,0>& g, const Array<T,1>& x,
    const U& i) {
  return single(g, i, length(x));
}

template<arithmetic T, scalar U>
NUMBIRCH_KEEP real element_grad2(const Array<real,0>& g, const Array<T,1>& x, const U& i) {
  return real(0);
}

template<arithmetic T, scalar U, scalar V>
NUMBIRCH_KEEP Array<T,0> element(const Array<T,2>& A, const U& i, const V& j) {
  Array<T,0> z;
  for_each(1, 1, gather_functor{buffer(A), stride(A), buffer(i), stride(i),
      buffer(j), stride(j), buffer(z), stride(z)});
  return z;
}

template<arithmetic T, scalar U, scalar V>
NUMBIRCH_KEEP Array<real,2> element_grad1(const Array<real,0>& g, const Array<T,2>& A,
    const U& i, const V& j) {
  return single(g, i, j, rows(A), columns(A));
}

template<arithmetic T, scalar U, scalar V>
NUMBIRCH_KEEP real element_grad2(const Array<real,0>& g, const Array<T,2>& A, const U& i,
    const V& j) {
  return real(0);
}

template<arithmetic T, scalar U, scalar V>
NUMBIRCH_KEEP real element_grad3(const Array<real,0>& g, const Array<T,2>& A, const U& i,
    const V& j) {
  return real(0);
}

template<scalar T, scalar U>
NUMBIRCH_KEEP Array<value_t<T>,1> single(const T& x, const U& i, const int n) {
  Array<value_t<T>,1> y(make_shape(n));
  for_each(n, single_functor{buffer(x), 1, buffer(i), buffer(y), stride(y)});
  return y;
}

template<scalar T, scalar U>
NUMBIRCH_KEEP Array<real,0> single_grad1(const Array<real,1>& g, const T& x, const U& i,
    const int n) {
  return element(g, i);
}

template<scalar T, scalar U>
NUMBIRCH_KEEP real single_grad2(const Array<real,1>& g, const T& x, const U& i,
    const int n) {
  return real(0);
}

template<scalar T, scalar U, scalar V>
NUMBIRCH_KEEP Array<value_t<T>,2> single(const T& x, const U& i, const V& j, const int m,
    const int n) {
  Array<value_t<T>,2> Y(make_shape(m, n));
  for_each(m, n, single_functor{buffer(x), buffer(i), buffer(j), buffer(Y),
      stride(Y)});
  return Y;
}

template<scalar T, scalar U, scalar V>
NUMBIRCH_KEEP Array<real,0> single_grad1(const Array<real,2>& g, const T& x, const U& i,
    const V& j, const int m, const int n) {
  return element(g, i, j);
}

template<scalar T, scalar U, scalar V>
NUMBIRCH_KEEP real single_grad2(const Array<real,2>& g, const T& x, const U& i, const V& j,
    const int m, const int n) {
  return real(0);
}

template<scalar T, scalar U, scalar V>
NUMBIRCH_KEEP real single_grad3(const Array<real,2>& g, const T& x, const U& i, const V& j,
    const int m, const int n) {
  return real(0);
}

template<numeric T, numeric U>
NUMBIRCH_KEEP pack_t<T,U> pack(const T& x, const U& y) {
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

template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<T> pack_grad1(const real_t<pack_t<T,U>>& g, const T& x, const U& y) {
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

template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<U> pack_grad2(const real_t<pack_t<T,U>>& g, const T& x, const U& y) {
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

template<numeric T, numeric U>
NUMBIRCH_KEEP stack_t<T,U> stack(const T& x, const U& y) {
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

template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<T> stack_grad1(const real_t<stack_t<T,U>>& g, const T& x, const U& y) {
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

template<numeric T, numeric U>
NUMBIRCH_KEEP real_t<U> stack_grad2(const real_t<stack_t<T,U>>& g, const T& x, const U& y) {
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

template<numeric T>
NUMBIRCH_KEEP Array<value_t<T>,0> scal(const T& x) {
  if constexpr (is_arithmetic_v<T>) {
    return x;
  } else {
    return x.scal();
  }
}

template<numeric T>
NUMBIRCH_KEEP real_t<T> scal_grad(const Array<real,0>& g, const T& x) {
  if constexpr (is_scalar_v<T>) {
    return g.scal();
  } else if constexpr (is_vector_v<T>) {
    return g.vec();
  } else {
    return g.mat(1);
  }
}

template<numeric T>
NUMBIRCH_KEEP Array<value_t<T>,1> vec(const T& x) {
  if constexpr (is_arithmetic_v<T>) {
    return x;
  } else if (x.contiguous() || is_vector_v<T>) {
    return x.vec();
  } else {
    Array<value_t<T>,1> y(make_shape(size(x)));
    for_each(size(x), reshape_functor{width(x), 1, buffer(x), stride(x),
        buffer(y), stride(y)});
    return y;
  }
}

template<numeric T>
NUMBIRCH_KEEP real_t<T> vec_grad(const Array<real,1>& g, const T& x) {
  if constexpr (is_scalar_v<T>) {
    return g.scal();
  } else if constexpr (is_vector_v<T>) {
    return g.vec();
  } else {
    return g.mat(columns(x));
  }
}

template<numeric T>
NUMBIRCH_KEEP Array<value_t<T>,2> mat(const T& x, const int n) {
  assert(size(x) % n == 0);
  if constexpr (is_arithmetic_v<T>) {
    return x;
  } else if (x.contiguous() || (is_matrix_v<T> && columns(x) == n)) {
    return x.mat(n);
  } else {
    Array<value_t<T>,2> y(make_shape(size(x)/n, n));
    for_each(size(x)/n, n, reshape_functor{width(x), size(x)/n, buffer(x),
        stride(x), buffer(y), stride(y)});
    return y;
  }
}

template<numeric T>
NUMBIRCH_KEEP real_t<T> mat_grad(const Array<real,2>& g, const T& x, const int n) {
  if constexpr (is_scalar_v<T>) {
    return g.scal();
  } else if constexpr (is_vector_v<T>) {
    return g.vec();
  } else {
    return g.mat(columns(x));
  }
}

template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<T,1> gather(const Array<T,1>& x, const Array<U,1>& y) {
  Array<T,1> z(shape(y));
  for_each(length(y), gather_functor{buffer(x), stride(x), 1, 0, buffer(y),
      stride(y), buffer(z), stride(z)});
  return z;
}

template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<real,1> gather_grad1(const Array<real,1>& g, const Array<T,1>& x,
    const Array<U,1>& y) {
  return scatter(g, y, length(x));
}

template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<real,1> gather_grad2(const Array<real,1>& g, const Array<T,1>& x,
    const Array<U,1>& y) {
  return fill(real(0), length(y));
}

template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<T,2> gather(const Array<T,2>& A, const Array<U,2>& I,
    const Array<V,2>& J) {
  Array<T,2> C(shape(I));
  for_each(rows(I), columns(I), gather_functor{buffer(A), stride(A),
      buffer(I), stride(I), buffer(J), stride(J), buffer(C), stride(C)});
  return C;
}

template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> gather_grad1(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J) {
  return scatter(G, I, J, rows(A), columns(A));
}

template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> gather_grad2(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J) {
  return fill(real(0), rows(I), columns(I));
}

template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> gather_grad3(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J) {
  return fill(real(0), rows(J), columns(J));
}

template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<T,1> scatter(const Array<T,1>& x, const Array<U,1>& y, const int n) {
  assert(conforms(x, y));
  auto z = fill(T(0), n);
  for_each(length(x), scatter_functor{buffer(x), stride(x), 1, 0, buffer(y),
      stride(y), buffer(z), stride(z)});
  return z;
}

template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<real,1> scatter_grad1(const Array<real,1>& g, const Array<T,1>& x,
    const Array<U,1>& y, const int n) {
  return gather(g, y);
}

template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP Array<real,1> scatter_grad2(const Array<real,1>& g, const Array<T,1>& x,
    const Array<U,1>& y, const int n) {
  return Array<real,1>(real(0), shape(y));
}

template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<T,2> scatter(const Array<T,2>& A, const Array<U,2>& I,
    const Array<V,2>& J, const int m, const int n) {
  assert(conforms(A, I));
  assert(conforms(A, J));
  auto C = fill(T(0), m, n);
  for_each(rows(A), columns(A), scatter_functor{buffer(A), stride(A),
      buffer(I), stride(I), buffer(J), stride(J), buffer(C), stride(C)});
  return C;
}

template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> scatter_grad1(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J, const int m, const int n) {
  return gather(G, I, J);
}

template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> scatter_grad2(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J, const int m, const int n) {
  return fill(real(0), rows(I), columns(I));
}

template<arithmetic T, arithmetic U, arithmetic V>
NUMBIRCH_KEEP Array<real,2> scatter_grad3(const Array<real,2>& G, const Array<T,2>& A,
    const Array<U,2>& I, const Array<V,2>& J, const int m, const int n) {
  return fill(real(0), rows(J), columns(J));
}

}
