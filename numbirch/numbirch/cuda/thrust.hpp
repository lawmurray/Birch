/**
 * @file
 * 
 * Thrust integration, including functors, iterators and allocators.
 *
 * Thrust support for lambda functions has some limitations as of CUDA 11.4.
 * In particular the nvcc ommand-line option --extended-lambda may be
 * necessary, and even with this the use of lambda functions within functions
 * with deduced return types is not supported. For this reason functors are
 * used instead of lambdas.
 */
#pragma once

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/async/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

template<class T>
struct VectorElementFunctor : public thrust::unary_function<int,T&> {
  VectorElementFunctor(T* x, int incx) :
      x(x),
      incx(incx) {
    //
  }
  __host__ __device__
  T& operator()(const int i) {
    return x[i*incx];
  }
  T* x;
  int incx;
};

template<class T>
struct MatrixElementFunctor : public thrust::unary_function<int,T&> {
  MatrixElementFunctor(T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  __host__ __device__
  T& operator()(const int i) {
    int c = i/m;
    int r = i - c*m;
    return A[r + c*ldA];
  }
  T* A;
  int m;
  int ldA;
};

template<class T>
struct MatrixTransposeElementFunctor : public thrust::unary_function<int,T&> {
  MatrixTransposeElementFunctor(T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  __host__ __device__
  T& operator()(const int i) {
    int r = i/m;
    int c = i - r*m;
    return A[r + c*ldA];
  }
  T* A;
  int m;
  int ldA;
};

template<class T>
struct MatrixLowerElementFunctor : public thrust::unary_function<int,T> {
  MatrixLowerElementFunctor(const T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  __host__ __device__
  T operator()(const int i) {
    int c = i/m;
    int r = i - c*m;
    return (c <= r) ? A[r + c*ldA] : 0.0;
  }
  const T* A;
  int m;
  int ldA;
};

template<class T>
struct MatrixSymmetricElementFunctor : public thrust::unary_function<int,T> {
  MatrixSymmetricElementFunctor(const T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  __host__ __device__
  T operator()(const int i) {
    int c = i/m;
    int r = i - c*m;
    return A[(c <= r) ? (r + c*ldA) : (c + r*ldA)];
  }
  const T* A;
  int m;
  int ldA;
};

template<class T = double>
struct ScalarMultiplyFunctor : public thrust::unary_function<T,T> {
  ScalarMultiplyFunctor(T a) :
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
struct ScalarDivideFunctor : public thrust::unary_function<T,T> {
  ScalarDivideFunctor(T a) :
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
struct LogAbsFunctor : public thrust::unary_function<T,T> {
  __host__ __device__
  T operator()(const T x) const {
    return std::log(std::abs(x));
  }
};

template<class T = double>
struct LogFunctor : public thrust::unary_function<T,T> {
  __host__ __device__
  T operator()(const T x) const {
    return std::log(x);
  }
};

/*
 * Simplified handling of pairs of iterators that define a range of a Thrust
 * vector or matrix.
 */

template<class Iterator>
struct ThrustRange {
  ThrustRange(Iterator first, Iterator second) :
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
static auto make_thrust_range(Iterator first, Iterator second) {
  return ThrustRange<Iterator>(first, second);
}

/*
 * Factory functions for creating Thrust vectors and matrices of various
 * types.
 */

template<class T>
static auto make_thrust_vector(T* x, const int n, const int incx) {
  auto begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    VectorElementFunctor<T>(x, incx));
  return make_thrust_range(begin, begin + n);
}

template<class T>
static auto make_thrust_matrix(T* A, const int m, const int n,
    const int ldA) {
  auto begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    MatrixElementFunctor<T>(A, m, ldA));
  return make_thrust_range(begin, begin + m*n);
}

template<class T>
static auto make_thrust_matrix_transpose(T* A, const int m, const int n,
    const int ldA) {
  auto begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    MatrixTransposeElementFunctor<T>(A, m, ldA));
  return make_thrust_range(begin, begin + m*n);
}

template<class T>
static auto make_thrust_matrix_lower(const T* A, const int m, const int n,
    const int ldA) {
  auto begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    MatrixLowerElementFunctor<T>(A, m, ldA));
  return make_thrust_range(begin, begin + m*n);
}

template<class T>
static auto make_thrust_matrix_symmetric(const T* A, const int m, const int n,
    const int ldA) {
  auto begin = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0),
    MatrixSymmetricElementFunctor<T>(A, m, ldA));
  return make_thrust_range(begin, begin + m*n);
}
