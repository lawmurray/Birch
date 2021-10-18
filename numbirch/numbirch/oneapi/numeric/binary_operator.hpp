/**
 * @file
 */
#pragma once

#include "numbirch/oneapi/sycl.hpp"
#include "numbirch/oneapi/mkl.hpp"
#include "numbirch/oneapi/dpl.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/functor.hpp"
#include "numbirch/memory.hpp"

namespace numbirch {

template<class T>
void add(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(),
      A1.end(), B1.begin(), C1.begin(), dpl::plus<T>());
}

template<class T, class U>
void div(const int m, const int n, const T* A, const int ldA, const U* b,
    T* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(),
      A1.end(), C1.begin(), divide_scalar_functor<T,U>(b));
}

template<class T, class U>
void mul(const int m, const int n, const T* a, const U* B, const int ldB,
    U* C, const int ldC) {
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(dpl::execution::make_device_policy(queue), B1.begin(),
      B1.end(), C1.begin(), multiply_scalar_functor<U,T>(a));
}

template<class T>
void mul(const int m, const int n, const T* A, const int ldA, const T* x,
    const int incx, T* y, const int incy) {
  blas::gemv(queue, mkl::transpose::N, m, n, 1.0, A, ldA, x, incx, 0.0, y,
      incy);
}

template<class T>
void mul(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  blas::gemm(queue, mkl::transpose::N, mkl::transpose::N, m, n, k, 1.0, A,
      ldA, B, ldB, 0.0, C, ldC);
}

template<class T>
void sub(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(),
      A1.end(), B1.begin(), C1.begin(), dpl::minus<T>());
}

}
