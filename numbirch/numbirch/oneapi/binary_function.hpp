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
void cholmul(const int n, const T* S, const int ldS, const T* x,
    const int incx, T* y, const int incy) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto L = (T*)device_malloc(n*n*sizeof(T));
  auto ldL = n;

  /* Cholesky factorization */
  blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  lapack::potrf(queue, mkl::uplo::lower, n, L, ldL, scratchpad,
      scratchpad_size);

  /* multiply */
  blas::copy(queue, n, x, incx, y, incy);
  blas::trmv(queue, mkl::uplo::lower, mkl::transpose::N, mkl::diag::N, n,
      L, ldL, y, incy);

  device_free(L);
  device_free(scratchpad);
}

template<class T>
void cholmul(const int m, const int n, const T* S, const int ldS, const T* B,
    const int ldB, T* C, const int ldC) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, m, ldS);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto L = (T*)device_malloc(m*m*sizeof(T));
  auto ldL = m;

  /* Cholesky factorization */
  blas::copy_batch(queue, m, S, 1, ldS, L, 1, ldL, m);
  lapack::potrf(queue, mkl::uplo::lower, m, L, ldL, scratchpad,
      scratchpad_size);

  /* multiply */
  blas::copy_batch(queue, m, B, 1, ldB, C, 1, ldC, n);
  blas::trmm(queue, mkl::side::left, mkl::uplo::lower, mkl::transpose::N,
      mkl::diag::N, m, n, 1.0, L, ldL, C, ldC);

  device_free(L);
  device_free(scratchpad);
}

template<class T>
void cholouter(const int m, const int n, const T* A, const int ldA,
    const T* S, const int ldS, T* C, const int ldC) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto L = (T*)device_malloc(n*n*sizeof(T));
  auto ldL = n;

  /* Cholesky factorization */
  blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  lapack::potrf(queue, mkl::uplo::lower, n, L, ldL, scratchpad,
      scratchpad_size);

  /* multiply */
  blas::copy_batch(queue, m, A, 1, ldA, C, 1, ldC, n);
  blas::trmm(queue, mkl::side::right, mkl::uplo::lower, mkl::transpose::T,
      mkl::diag::N, m, n, 1.0, L, ldL, C, ldC);

  device_free(L);
  device_free(scratchpad);
}

template<class T>
void cholsolve(const int n, const T* S, const int ldS, T* x, const int incx,
    const T* y, const int incy) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto L = (T*)device_malloc(n*n*sizeof(T));
  auto ldL = n;
  T* x1 = x;
  if (incx > 1) {
    x1 = (T*)device_malloc(n*sizeof(T));
  }
  int incx1 = 1;

  /* solve via Cholesky factorization */
  blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  lapack::potrf(queue, mkl::uplo::lower, n, L, ldL, scratchpad,
      scratchpad_size);
  blas::copy(queue, n, y, incy, x1, incx1);
  lapack::potrs(queue, mkl::uplo::lower, n, 1, L, ldL, x1, n, scratchpad,
      scratchpad_size);
  if (incx > 1) {
    blas::copy(queue, n, x1, incx1, x, incx);
    device_free(x1);
  }

  device_free(L);
  device_free(scratchpad);
}

template<class T>
void cholsolve(const int m, const int n, const T* S, const int ldS, T* X,
    const int ldX, const T* Y, const int ldY) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, m, ldS);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto L = (T*)device_malloc(m*m*sizeof(T));
  auto ldL = m;

  /* solve via Cholesky factorization */
  blas::copy_batch(queue, m, S, 1, ldS, L, 1, ldL, m);
  lapack::potrf(queue, mkl::uplo::lower, m, L, ldL, scratchpad,
      scratchpad_size);
  blas::copy_batch(queue, m, Y, 1, ldY, X, 1, ldX, n);
  lapack::potrs(queue, mkl::uplo::lower, m, n, L, ldL, X, ldX, scratchpad,
      scratchpad_size);

  device_free(L);
  device_free(scratchpad);
}

template<class T>
void diagonal(const T* a, const int n, T* B, const int ldB) {
  ///@todo Implement as single kernel
  auto B1 = make_dpl_matrix(B, n, n, ldB);
  auto d = make_dpl_vector(B, n, ldB + 1);  // diagonal
  dpl::experimental::fill_async(dpl::execution::make_device_policy(queue),
      B1.begin(), B1.end(), 0.0);
  dpl::experimental::fill_async(dpl::execution::make_device_policy(queue),
      d.begin(), d.end(), *a);
}

template<class T>
T dot(const int n, const T* x, const int incx, const T* y, const int incy) {
  auto z = (T*)device_malloc(sizeof(T));
  blas::dot(queue, n, x, incx, y, incy, z);
  wait();
  auto res = z[0];
  device_free(z);
  return res;
}

template<class T>
T frobenius(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  return dpl::transform_reduce(dpl::execution::make_device_policy(queue),
      A1.begin(), A1.end(), B1.begin(), 0.0, dpl::plus<T>(),
      dpl::multiplies<T>());
}

template<class T>
void hadamard(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(),
      A1.end(), B1.begin(), C1.begin(), dpl::multiplies<T>());
}

template<class T>
void inner(const int m, const int n, const T* A, const int ldA, const T* x,
    const int incx, T* y, const int incy) {
  blas::gemv(queue, mkl::transpose::T, n, m, 1.0, A, ldA, x, incx, 0.0, y,
      incy);
}

template<class T>
void inner(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  blas::gemm(queue, mkl::transpose::T, mkl::transpose::N, m, n, k, 1.0, A,
      ldA, B, ldB, 0.0, C, ldC);
}

template<class T>
void outer(const int m, const int n, const T* x, const int incx, const T* y,
    const int incy, T* A, const int ldA) {
  /* here, the two vectors are interpreted as single-row matrices, so that the
   * stride between elements becomes the stride between columns; to create the
   * outer product, the first matrix is transposed to a single-column matrix,
   * while the second is not */
  blas::gemm(queue, mkl::transpose::T, mkl::transpose::N, m, n, 1, 1.0, x,
      incx, y, incy, 0.0, A, ldA);
}

template<class T>
void outer(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  blas::gemm(queue, mkl::transpose::N, mkl::transpose::T, m, n, k, 1.0, A,
      ldA, B, ldB, 0.0, C, ldC);
}

template<class T>
void solve(const int n, const T* A, const int ldA, T* x, const int incx,
    const T* y, const int incy) {
  auto scratchpad_size = lapack::getrf_scratchpad_size<T>(queue, n, n, ldA);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto ipiv = (int64_t*)device_malloc(std::max(1, n)*sizeof(int64_t));
  auto L = (T*)device_malloc(std::max(1, n*n)*sizeof(T));
  auto ldL = n;
  T* x1 = x;
  if (incx > 1) {
    x1 = (T*)device_malloc(n*sizeof(T));
  }

  /* solve via L factorization with partial pivoting */
  blas::copy_batch(queue, n, A, 1, ldA, L, 1, ldL, n);
  lapack::getrf(queue, n, n, L, ldL, ipiv, scratchpad, scratchpad_size);
  blas::copy(queue, n, y, incy, x1, 1);
  lapack::getrs(queue, mkl::transpose::N, n, n, L, ldL, ipiv, x1, n,
      scratchpad, scratchpad_size);
  if (incx > 1) {
    blas::copy(queue, n, x1, 1, x, incx);
    device_free(x1);
  }

  device_free(L);
  device_free(ipiv);
  device_free(scratchpad);
}

template<class T>
void solve(const int m, const int n, const T* A, const int ldA, T* X,
    const int ldX, const T* Y, const int ldY) {
  auto scratchpad_size = lapack::getrf_scratchpad_size<T>(queue, m,
      m, ldA);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto ipiv = (int64_t*)device_malloc(std::max(1, m)*sizeof(int64_t));
  auto L = (T*)device_malloc(std::max(1, m*m)*sizeof(T));
  auto ldL = m;

  /* solve via L factorization with partial pivoting */
  blas::copy_batch(queue, m, A, 1, ldA, L, 1, ldL, m);
  lapack::getrf(queue, m, m, L, ldL, ipiv, scratchpad, scratchpad_size);
  blas::copy_batch(queue, m, Y, 1, ldY, X, 1, ldX, n);
  lapack::getrs(queue, mkl::transpose::N, m, n, L, ldL, ipiv, X, ldX,
      scratchpad, scratchpad_size);

  device_free(L);
  device_free(ipiv);
  device_free(scratchpad);
}

}
