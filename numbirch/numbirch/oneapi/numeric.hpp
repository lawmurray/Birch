/**
 * @file
 */
#pragma once

#include "numbirch/oneapi/sycl.hpp"
#include "numbirch/oneapi/mkl.hpp"
#include "numbirch/oneapi/dpl.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/common/functor.hpp"
#include "numbirch/memory.hpp"

namespace numbirch {

void logical_not(const int m, const int n, const bool* A, const int ldA,
    bool* B, const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(),
      A1.end(), B1.begin(), logical_not_functor());
}

template<class T>
void neg(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(),
      A1.end(), B1.begin(), dpl::negate<T>());
}

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

template<class T>
void cholinv(const int n, const T* S, const int ldS, T* B, const int ldB) {
  auto scratchpad_size1 = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldB);
  auto scratchpad_size2 = lapack::potri_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldB);
  auto scratchpad_size = std::max(scratchpad_size1, scratchpad_size2);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto A = (T*)device_malloc(n*n*sizeof(T));
  auto ldA = n;

  /* invert via Cholesky factorization */
  blas::copy_batch(queue, n, S, 1, ldS, A, 1, ldA, n);
  lapack::potrf(queue, mkl::uplo::lower, n, A, ldA, scratchpad,
      scratchpad_size);
  lapack::potri(queue, mkl::uplo::lower, n, A, ldA, scratchpad,
      scratchpad_size);

  /* potri only modifies the lower triangle of A, whereas caller expects the
   * whole symmetric matrix (at least for now); copy that into B */
  auto A1 = make_dpl_matrix_symmetric(A, n, n, ldA);
  auto B1 = make_dpl_matrix(B, n, n, ldB);
  dpl::copy(dpl::execution::make_device_policy(queue), A1.begin(), A1.end(),
      B1.begin());

  device_free(A);
  device_free(scratchpad);
}

template<class T>
T count(const int m, const int n, const T* A, const int ldA) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  return dpl::transform_reduce(dpl::execution::make_device_policy(queue),
      A1.begin(), A1.end(), 0, dpl::plus<int>(), count_functor<T>());
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
void inv(const int n, const T* A, const int ldA, T* B, const int ldB) {
  auto scratchpad_size1 = lapack::getrf_scratchpad_size<T>(queue, n,
      n, ldB);
  auto scratchpad_size2 = lapack::getri_scratchpad_size<T>(queue, n,
      ldB);
  auto scratchpad_size = std::max(scratchpad_size1, scratchpad_size2);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto ipiv = (int64_t*)device_malloc(std::max(1, n)*sizeof(int64_t));

  /* invert via L factorization with partial pivoting */
  blas::copy_batch(queue, n, A, 1, ldA, B, 1, ldB, n);
  lapack::getrf(queue, n, n, B, ldB, ipiv, scratchpad, scratchpad_size);
  lapack::getri(queue, n, B, ldB, ipiv, scratchpad, scratchpad_size);

  device_free(ipiv);
  device_free(scratchpad);
}

template<class T>
T lcholdet(const int n, const T* S, const int ldS) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto L = (T*)device_malloc(n*n*sizeof(T));
  auto ldL = n;

  /* Cholesky factorization */
  blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  lapack::potrf(queue, mkl::uplo::lower, n, L, ldL, scratchpad,
      scratchpad_size);

  /* log-determinant is twice the sum of logarithms of elements on the main
   * diagonal, all of which should be positive; the 2.0 is multiplied in by
   * the return statement below */
  auto d = make_dpl_vector(L, n, ldL + 1);  // diagonal of L
  auto half_ldet = dpl::experimental::transform_reduce_async(
      dpl::execution::make_device_policy(queue), d.begin(), d.end(), 0.0,
      dpl::plus<T>(), log_functor<T>());
  wait();

  device_free(L);
  device_free(scratchpad);

  return 2.0*half_ldet.get();
}

template<class T>
T ldet(const int n, const T* A, const int ldA) {
  auto scratchpad_size = lapack::getrf_scratchpad_size<T>(queue, n, n,
      ldA);
  auto scratchpad = (T*)device_malloc(scratchpad_size*sizeof(T));
  auto ipiv = (int64_t*)device_malloc(std::max(1, n)*sizeof(int64_t));
  auto L = (T*)device_malloc(std::max(1, n*n)*sizeof(T));
  auto ldL = n;

  /* L factorization with partial pivoting */
  blas::copy_batch(queue, n, A, 1, ldA, L, 1, ldL, n);
  lapack::getrf(queue, n, n, L, ldL, ipiv, scratchpad, scratchpad_size);

  /* the L factorization is with partial pivoting, which means $|A| = (-1)^p
   * |L||U|$, where $p$ is the number of row exchanges in `ipiv`; however,
   * we're taking the logarithm of its absolute value, so can ignore the first
   * term, and the second term is just 1 as $L$ has a unit diagonal; just need
   * $|U|$ here; the logarithm of its absolute value is just the sum of the
   * logarithms of the absolute values of elements on the main diagonal */
  auto d = make_dpl_vector(L, n, ldL + 1);  // diagonal of L
  auto ldet = dpl::experimental::transform_reduce_async(
      dpl::execution::make_device_policy(queue), d.begin(), d.end(), 0.0,
      dpl::plus<T>(), log_abs_functor<T>());
  wait();

  device_free(L);
  device_free(ipiv);
  device_free(scratchpad);

  return ldet.get();
}

template<class T>
void rcp(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(),
      A1.end(), B1.begin(), rcp_functor<T>());
}

template<class T>
void rectify(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(),
      A1.end(), B1.begin(), rectify_functor<T>());
}

template<class T>
void single(const int* i, const int n, T* x, const int incx) {
  ///@todo Implement as single kernel
  auto x1 = make_dpl_vector(x, n, incx);
  dpl::experimental::fill_async(dpl::execution::make_device_policy(queue),
      x1.begin(), x1.end(), 0.0);
  *(x1.begin() + *i) = T(1);
}

template<class T>
T sum(const int m, const int n, const T* A, const int ldA) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  return dpl::reduce(dpl::execution::make_device_policy(queue), A1.begin(),
     A1.end());
}

template<class T>
T trace(const int m, const int n, const T* A, const int ldA) {
  return sum(1, std::min(m, n), A, ldA + 1);
}

template<class T>
void transpose(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_dpl_matrix_transpose(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::copy(dpl::execution::make_device_policy(queue),
      A1.begin(), A1.end(), B1.begin());
}

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
