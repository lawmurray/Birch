/**
 * @file
 * 
 * CUB integration.
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"

#include <cub/cub.cuh>

namespace numbirch {
template<class T>
struct vector_element_functor {
  vector_element_functor(T* x, int incx) :
      x(x),
      incx(incx) {
    //
  }
  __host__ __device__
  T operator()(const int i) const {
    return x[i*incx];
  }
  T* x;
  int incx;
};

template<class T>
struct matrix_element_functor {
  matrix_element_functor(T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  __host__ __device__
  T operator()(const int i) const {
    int c = i/m;
    int r = i - c*m;
    return A[r + c*ldA];
  }
  T* A;
  int m;
  int ldA;
};

template<class T>
struct matrix_transpose_element_functor {
  matrix_transpose_element_functor(T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  __host__ __device__
  T operator()(const int i) const {
    int r = i/m;
    int c = i - r*m;
    return A[r + c*ldA];
  }
  T* A;
  int m;
  int ldA;
};

template<class T = double>
struct negate_functor {
  __host__ __device__
  T operator()(const T x) const {
    return -x;
  }
};

template<class T = double>
struct plus_functor {
  __host__ __device__
  T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<class T = double>
struct minus_functor {
  __host__ __device__
  T operator()(const T x, const T y) const {
    return x - y;
  }
};

template<class T = double>
struct multiplies_functor {
  __host__ __device__
  T operator()(const T x, const T y) const {
    return x*y;
  }
};

template<class T = double>
struct divides_functor {
  __host__ __device__
  T operator()(const T x, const T y) const {
    return x/y;
  }
};

template<class T = double>
struct scalar_multiplies_functor {
  scalar_multiplies_functor(T a) :
      a(a) {
    //
  }
  __host__ __device__
  T operator()(const T x) const {
    return x*a;
  }
  T a;
};

template<class T = double>
struct scalar_divides_functor {
  scalar_divides_functor(T a) :
      a(a) {
    //
  }
  __host__ __device__
  T operator()(const T x) const {
    return x/a;
  }
  T a;
};

template<class T = double>
struct log_abs_functor {
  __host__ __device__
  T operator()(const T x) const {
    return std::log(std::abs(x));
  }
};

template<class T = double>
struct log_functor {
  __host__ __device__
  T operator()(const T x) const {
    return std::log(x);
  }
};

/*
 * Pair of iterators defining a range.
 */
template<class Iterator>
struct cub_range {
  cub_range(Iterator first, Iterator second) :
      first(first),
      second(second) {
    //
  }
  auto begin() {
    return first;
  }
  auto end() {
    return second;
  }
  Iterator first;
  Iterator second;
};

template<class Iterator>
static auto make_cub_range(Iterator first, Iterator second) {
  return cub_range<Iterator>(first, second);
}

template<class T>
static auto make_cub_vector(T* x, const int n, const int incx) {
  auto elem = vector_element_functor<T>(x, incx);
  auto count = cub::CountingInputIterator<int>(0);
  auto begin = cub::TransformInputIterator<T,decltype(elem),decltype(count)>(
      count, elem);
  return make_cub_range(begin, begin + n);
}

template<class T>
static auto make_cub_matrix(T* A, const int m, const int n, const int ldA) {
  auto elem = matrix_element_functor<T>(A, m, ldA);
  auto count = cub::CountingInputIterator<int>(0);
  auto begin = cub::TransformInputIterator<T,decltype(elem),decltype(count)>(
      count, elem);
  return make_cub_range(begin, begin + m*n);
}

template<class T>
static auto make_cub_matrix_transpose(T* A, const int m, const int n,
    const int ldA) {
  auto elem = matrix_transpose_element_functor<T>(A, m, ldA);
  auto count = cub::CountingInputIterator<int>(0);
  auto begin = cub::TransformInputIterator<T,decltype(elem),decltype(count)>(
      count, elem);
  return make_cub_range(begin, begin + m*n);
}

}
