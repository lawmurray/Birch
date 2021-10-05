/**
 * @file
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/cub.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/functor.hpp"
#include "numbirch/memory.hpp"
#include "numbirch/numeric.hpp"

namespace numbirch {

template<class T>
void add(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, add_functor<T>());
}

template<class T, class U>
void div(const int m, const int n, const T* A, const int ldA, const U* b,
    T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, C, ldC, divide_scalar_functor<T,U>(b));
}

template<class T>
void equal(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, bool* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, equal_functor<T>());
}

template<class T>
void greater(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, bool* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, greater_functor<T>());
}

template<class T>
void greater_or_equal(const int m, const int n, const T* A, const int ldA,
    const T* B, const int ldB, bool* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, greater_or_equal_functor<T>());
}


template<class T>
void less(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, bool* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, less_functor<T>());
}

template<class T>
void less_or_equal(const int m, const int n, const T* A, const int ldA,
    const T* B, const int ldB, bool* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, less_or_equal_functor<T>());
}

void logical_and(const int m, const int n, const bool* A, const int ldA,
    const bool* B, const int ldB, bool* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, logical_and_functor());
}

void logical_or(const int m, const int n, const bool* A, const int ldA,
    const bool* B, const int ldB, bool* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, logical_or_functor());
}

template<class T, class U>
void mul(const int m, const int n, const T* a, const U* B, const int ldB,
    U* C, const int ldC) {
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, B, ldB, C, ldC, multiply_scalar_functor<U,T>(a));
}

template<class T>
void mul(const int m, const int n, const T* A, const int ldA, const T* x,
    const int incx, T* y, const int incy) {
  prefetch(A, m, n, ldA);
  prefetch(x, n, incx);
  prefetch(y, m, incy);
  CUBLAS_CHECK(cublas<T>::gemv(cublasHandle, CUBLAS_OP_N, m, n,
      scalar<T>::one, A, ldA, x, incx, scalar<T>::zero, y, incy));
}

template<class T>
void mul(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  prefetch(A, m, k, ldA);
  prefetch(B, k, n, ldB);
  prefetch(C, m, n, ldC);
  CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
      k, scalar<T>::one, A, ldA, B, ldB, scalar<T>::zero, C, ldC));
}

template<class T>
void not_equal(const int m, const int n, const T* A, const int ldA,
    const T* B, const int ldB, bool* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, not_equal_functor<T>());
}

template<class T>
void sub(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, subtract_functor<T>());
}

}
