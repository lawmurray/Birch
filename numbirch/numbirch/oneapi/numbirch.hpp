/**
 * @file
 * 
 * oneAPI implementation of generic functions.
 */
#pragma once

#include "numbirch/oneapi/sycl.hpp"
#include "numbirch/oneapi/mkl.hpp"
#include "numbirch/oneapi/dpl.hpp"

namespace numbirch {

template<class T>
void neg(const int n, const T* x, const int incx, T* y, const int incy) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  dpl::transform(dpl::execution::make_device_policy(queue), x1.begin(), x1.end(), y1.begin(), dpl::negate<T>());
}

template<class T>
void neg(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(), A1.end(), B1.begin(), dpl::negate<T>());
}

template<class T>
void add(const int n, const T* x, const int incx, const T* y, const int incy,
    T* z, const int incz) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  auto z1 = make_dpl_vector(z, n, incz);
  dpl::transform(dpl::execution::make_device_policy(queue), x1.begin(), x1.end(), y1.begin(), z1.begin(),
      dpl::plus<T>());
}

template<class T>
void add(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(), A1.end(), B1.begin(), C1.begin(),
      dpl::plus<T>());
}

template<class T>
void sub(const int n, const T* x, const int incx, const T* y, const int incy,
    T* z, const int incz) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  auto z1 = make_dpl_vector(z, n, incz);
  dpl::transform(dpl::execution::make_device_policy(queue), x1.begin(), x1.end(), y1.begin(), z1.begin(),
      dpl::minus<T>());
}

template<class T>
void sub(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(), A1.end(), B1.begin(), C1.begin(),
      dpl::minus<T>());
}

template<class T>
void hadamard(const int n, const T* x, const int incx, const T* y,
    const int incy, T* z, const int incz) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  auto z1 = make_dpl_vector(z, n, incz);
  dpl::transform(dpl::execution::make_device_policy(queue), x1.begin(), x1.end(), y1.begin(), z1.begin(),
      dpl::multiplies<T>());
}

template<class T>
void hadamard(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(), A1.end(), B1.begin(), C1.begin(),
      dpl::multiplies<T>());
}

template<class T>
void div(const int n, const T* x, const int incx, const T y, T* z,
    const int incz) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto z1 = make_dpl_vector(z, n, incz);
  dpl::transform(dpl::execution::make_device_policy(queue), x1.begin(), x1.end(), z1.begin(), [=](T x) {
        return x/y; });
}

template<class T>
void div(const int m, const int n, const T* A, const int ldA, const T b, T* C,
    const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(), A1.end(), C1.begin(), [=](T a) {
        return a/b; });
}

template<class T>
void mul(const int n, const T x, const T* y, const int incy, T* z,
    const int incz) {
  auto evt = blas::axpby(queue, n, x, y, incy, 0.0, z, incz);
  evt.wait();
}

template<class T>
void mul(const int m, const int n, const T a, const T* B, const int ldB, T* C,
    const int ldC) {
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(dpl::execution::make_device_policy(queue), B1.begin(), B1.end(), C1.begin(), [=](T b) {
        return a*b; });
}

template<class T>
void mul(const int m, const int n, const T* A, const int ldA, const T* x,
    const int incx, T* y, const int incy) {
  auto evt = blas::gemv(queue, mkl::transpose::N, m, n, 1.0, A, ldA, x,
      incx, 0.0, y, incy);
  evt.wait();
}

template<class T>
void mul(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  auto evt = blas::gemm(queue, mkl::transpose::N, mkl::transpose::N, m,
      n, k, 1.0, A, ldA, B, ldB, 0.0, C, ldC);
  evt.wait();
}

template<class T>
void cholmul(const int n, const T* S, const int ldS, const T* x,
    const int incx, T* y, const int incy) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<T>(n*n, queue);
  auto ldL = n;

  /* Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, n, L, ldL,
      scratchpad, scratchpad_size, {evt1});

  /* multiply */
  auto evt3 = blas::copy(queue, n, x, incx, y, incy);
  auto evt4 = blas::trmv(queue, mkl::uplo::lower, mkl::transpose::N,
      mkl::diag::N, n, L, ldL, y, incy, {evt2, evt3});
  evt4.wait();

  sycl::free(L, queue);
  sycl::free(scratchpad, queue);
}

template<class T>
void cholmul(const int m, const int n, const T* S, const int ldS, const T* B,
    const int ldB, T* C, const int ldC) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, m, ldS);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<T>(m*m, queue);
  auto ldL = m;

  /* Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, m, S, 1, ldS, L, 1, ldL, m);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, m, L, ldL,
      scratchpad, scratchpad_size, {evt1});

  /* multiply */
  auto evt3 = blas::copy_batch(queue, m, B, 1, ldB, C, 1, ldC, n);
  auto evt4 = blas::trmm(queue, mkl::side::left, mkl::uplo::lower,
      mkl::transpose::N, mkl::diag::N, m, n, 1.0, L, ldL, C, ldC,
      {evt2, evt3});
  evt4.wait();

  sycl::free(L, queue);
  sycl::free(scratchpad, queue);
}

template<class T>
T sum(const int n, const T* x, const int incx) {
  auto x1 = make_dpl_vector(x, n, incx);
  return dpl::reduce(dpl::execution::make_device_policy(queue), x1.begin(), x1.end());
}

template<class T>
T sum(const int m, const int n, const T* A, const int ldA) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  return dpl::reduce(dpl::execution::make_device_policy(queue), A1.begin(), A1.end());
}

template<class T>
T dot(const int n, const T* x, const int incx, const T* y, const int incy) {
  auto z = sycl::malloc_shared<T>(1, queue);
  auto evt = blas::dot(queue, n, x, incx, y, incy, z);
  evt.wait();
  auto res = z[0];
  sycl::free(z, queue);
  return res;
}

template<class T>
T frobenius(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  return dpl::transform_reduce(dpl::execution::make_device_policy(queue), A1.begin(), A1.end(),
      B1.begin(), 0.0, dpl::plus<T>(), dpl::multiplies<T>());
}

template<class T>
void inner(const int m, const int n, const T* A, const int ldA, const T* x,
    const int incx, T* y, const int incy) {
  auto evt = blas::gemv(queue, mkl::transpose::T, n, m, 1.0, A, ldA, x,
      incx, 0.0, y, incy);
  evt.wait();
}

template<class T>
void inner(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  auto evt = blas::gemm(queue, mkl::transpose::T, mkl::transpose::N, m,
      n, k, 1.0, A, ldA, B, ldB, 0.0, C, ldC);
  evt.wait();
}

template<class T>
void outer(const int m, const int n, const T* x, const int incx, const T* y,
    const int incy, T* A, const int ldA) {
  /* here, the two vectors are interpreted as single-row matrices, so that the
   * stride between elements becomes the stride between columns; to create the
   * outer product, the first matrix is transposed to a single-column matrix,
   * while the second is not */
  auto evt = blas::gemm(queue, mkl::transpose::T, mkl::transpose::N, m,
      n, 1, 1.0, x, incx, y, incy, 0.0, A, ldA);
  evt.wait();
}

template<class T>
void outer(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  auto evt = blas::gemm(queue, mkl::transpose::N, mkl::transpose::T, m,
      n, k, 1.0, A, ldA, B, ldB, 0.0, C, ldC);
  evt.wait();
}

template<class T>
void cholouter(const int m, const int n, const T* A, const int ldA,
    const T* S, const int ldS, T* C, const int ldC) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<T>(n*n, queue);
  auto ldL = n;

  /* Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, n, L, ldL,
      scratchpad, scratchpad_size, {evt1});

  /* multiply */
  auto evt3 = blas::copy_batch(queue, m, A, 1, ldA, C, 1, ldC, n);
  auto evt4 = blas::trmm(queue, mkl::side::right, mkl::uplo::lower,
      mkl::transpose::T, mkl::diag::N, m, n, 1.0, L, ldL, C, ldC,
      {evt2, evt3});
  evt4.wait();

  sycl::free(L, queue);
  sycl::free(scratchpad, queue);
}

template<class T>
void solve(const int n, const T* A, const int ldA, T* x, const int incx,
    const T* y, const int incy) {
  auto scratchpad_size = lapack::getrf_scratchpad_size<T>(queue, n, n, ldA);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto ipiv = sycl::malloc_shared<int64_t>(std::max(1, n), queue);
  auto L = sycl::malloc_shared<T>(std::max(1, n*n), queue);
  auto ldL = n;
  T* x1 = x;
  if (incx > 1) {
    x1 = sycl::malloc_shared<T>(n, queue);
  }

  /* solve via L factorization with partial pivoting */
  auto evt1 = blas::copy_batch(queue, n, A, 1, ldA, L, 1, ldL, n);
  auto evt2 = lapack::getrf(queue, n, n, L, ldL, ipiv, scratchpad,
      scratchpad_size, {evt1});
  auto evt3 = blas::copy(queue, n, y, incy, x1, 1);
  auto evt4 = lapack::getrs(queue, mkl::transpose::N, n, n, L, ldL,
      ipiv, x1, n, scratchpad, scratchpad_size, {evt2, evt3});
  if (incx > 1) {
    auto evt5 = blas::copy(queue, n, x1, 1, x, incx, {evt4});
    evt5.wait();
    sycl::free(x1, queue);
  } else {
    evt4.wait();
  }

  sycl::free(L, queue);
  sycl::free(ipiv, queue);
  sycl::free(scratchpad, queue);
}

template<class T>
void solve(const int m, const int n, const T* A, const int ldA, T* X,
    const int ldX, const T* Y, const int ldY) {
  auto scratchpad_size = lapack::getrf_scratchpad_size<T>(queue, m,
      m, ldA);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto ipiv = sycl::malloc_shared<int64_t>(std::max(1, m), queue);
  auto L = sycl::malloc_shared<T>(std::max(1, m*m), queue);
  auto ldL = m;

  /* solve via L factorization with partial pivoting */
  auto evt1 = blas::copy_batch(queue, m, A, 1, ldA, L, 1, ldL, m);
  auto evt2 = lapack::getrf(queue, m, m, L, ldL, ipiv, scratchpad,
      scratchpad_size, {evt1});
  auto evt3 = blas::copy_batch(queue, m, Y, 1, ldY, X, 1, ldX, n);
  auto evt4 = lapack::getrs(queue, mkl::transpose::N, m, n, L, ldL,
      ipiv, X, ldX, scratchpad, scratchpad_size, {evt2, evt3});
  evt4.wait();

  sycl::free(L, queue);
  sycl::free(ipiv, queue);
  sycl::free(scratchpad, queue);
}

template<class T>
void cholsolve(const int n, const T* S, const int ldS, T* x, const int incx,
    const T* y, const int incy) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<T>(n*n, queue);
  auto ldL = n;
  T* x1 = x;
  if (incx > 1) {
    x1 = sycl::malloc_shared<T>(n, queue);
  }
  int incx1 = 1;

  /* solve via Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, n, L, ldL,
      scratchpad, scratchpad_size, {evt1});
  auto evt3 = blas::copy(queue, n, y, incy, x1, incx1);
  auto evt4 = lapack::potrs(queue, mkl::uplo::lower, n, 1, L, ldL,
      x1, n, scratchpad, scratchpad_size, {evt2, evt3});
  if (incx > 1) {
    auto evt5 = blas::copy(queue, n, x1, incx1, x, incx, {evt4});
    evt5.wait();
    sycl::free(x1, queue);
  } else {
    evt4.wait();
  }

  sycl::free(L, queue);
  sycl::free(scratchpad, queue);
}

template<class T>
void cholsolve(const int m, const int n, const T* S, const int ldS, T* X,
    const int ldX, const T* Y, const int ldY) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, m, ldS);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<T>(m*m, queue);
  auto ldL = m;

  /* solve via Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, m, S, 1, ldS, L, 1, ldL, m);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, m, L, ldL,
      scratchpad, scratchpad_size, {evt1});
  auto evt3 = blas::copy_batch(queue, m, Y, 1, ldY, X, 1, ldX, n);
  auto evt4 = lapack::potrs(queue, mkl::uplo::lower, m, n, L, ldL, X,
      ldX, scratchpad, scratchpad_size, {evt2, evt3});
  evt4.wait();

  sycl::free(L, queue);
  sycl::free(scratchpad, queue);
}

template<class T>
void inv(const int n, const T* A, const int ldA, T* B, const int ldB) {
  auto scratchpad_size1 = lapack::getrf_scratchpad_size<T>(queue, n,
      n, ldB);
  auto scratchpad_size2 = lapack::getri_scratchpad_size<T>(queue, n,
      ldB);
  auto scratchpad_size = std::max(scratchpad_size1, scratchpad_size2);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto ipiv = sycl::malloc_shared<int64_t>(std::max(1, n), queue);

  /* invert via L factorization with partial pivoting */
  auto evt1 = blas::copy_batch(queue, n, A, 1, ldA, B, 1, ldB, n);
  auto evt2 = lapack::getrf(queue, n, n, B, ldB, ipiv, scratchpad,
      scratchpad_size, {evt1});
  auto evt3 = lapack::getri(queue, n, B, ldB, ipiv, scratchpad,
      scratchpad_size, {evt2});
  evt3.wait();

  sycl::free(ipiv, queue);
  sycl::free(scratchpad, queue);
}

template<class T>
void cholinv(const int n, const T* S, const int ldS, T* B, const int ldB) {
  auto scratchpad_size1 = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldB);
  auto scratchpad_size2 = lapack::potri_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldB);
  auto scratchpad_size = std::max(scratchpad_size1, scratchpad_size2);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto A = sycl::malloc_shared<T>(n*n, queue);
  auto ldA = n;

  /* invert via Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, A, 1, ldA, n);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, n, A, ldA,
      scratchpad, scratchpad_size, {evt1});
  auto evt3 = lapack::potri(queue, mkl::uplo::lower, n, A, ldA,
      scratchpad, scratchpad_size, {evt2});

  /* potri only modifies the lower triangle of A, whereas caller expects the
   * whole symmetric matrix (at least for now); copy that into B */
  auto A1 = make_dpl_matrix_symmetric(A, n, n, ldA);
  auto B1 = make_dpl_matrix(B, n, n, ldB);
  // auto evt4 = dpl::experimental::copy_async(dpl::execution::make_device_policy(queue), A1.begin(),
  //    A1.end(), B1.begin(), evt3);
  // evt4.wait();
  // ^ compile errors as of Intel oneDPL 2021.4.0
  evt3.wait();
  dpl::copy(dpl::execution::make_device_policy(queue), A1.begin(), A1.end(), B1.begin());

  sycl::free(A, queue);
  sycl::free(scratchpad, queue);
}

template<class T>
T ldet(const int n, const T* A, const int ldA) {
  auto scratchpad_size = lapack::getrf_scratchpad_size<T>(queue, n,
      n, ldA);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto ipiv = sycl::malloc_shared<int64_t>(std::max(1, n), queue);
  auto L = sycl::malloc_shared<T>(std::max(1, n*n), queue);
  auto ldL = n;

  /* L factorization with partial pivoting */
  auto evt1 = blas::copy_batch(queue, n, A, 1, ldA, L, 1, ldL, n);
  auto evt2 = lapack::getrf(queue, n, n, L, ldL, ipiv, scratchpad,
      scratchpad_size, {evt1});

  /* the L factorization is with partial pivoting, which means $|A| = (-1)^p
   * |L||U|$, where $p$ is the number of row exchanges in `ipiv`; however,
   * we're taking the logarithm of its absolute value, so can ignore the first
   * term, and the second term is just 1 as $L$ has a unit diagonal; just need
   * $|U|$ here; the logarithm of its absolute value is just the sum of the
   * logarithms of the absolute values of elements on the main diagonal */
  auto d = make_dpl_vector(L, n, ldL + 1);  // diagonal of L
  auto logabs = [](T x) { return std::log(std::abs(x)); };
  auto ldet = dpl::experimental::transform_reduce_async(dpl::execution::make_device_policy(queue),
      d.begin(), d.end(), 0.0, dpl::plus<T>(), logabs, evt2);
  ldet.wait();

  sycl::free(L, queue);
  sycl::free(ipiv, queue);
  sycl::free(scratchpad, queue);

  return ldet.get();
}

template<class T>
T lcholdet(const int n, const T* S, const int ldS) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<T>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = sycl::malloc_shared<T>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<T>(n*n, queue);
  auto ldL = n;

  /* Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, n, L, ldL,
      scratchpad, scratchpad_size, {evt1});

  /* log-determinant is twice the sum of logarithms of elements on the main
   * diagonal, all of which should be positive; the 2.0 is multiplied in by
   * the return statement below */
  auto d = make_dpl_vector(L, n, ldL + 1);  // diagonal of L
  auto log = [](T x) { return std::log(x); };
  auto half_ldet = dpl::experimental::transform_reduce_async(dpl::execution::make_device_policy(queue),
      d.begin(), d.end(), 0.0, dpl::plus<T>(), log, evt2);
  half_ldet.wait();

  sycl::free(L, queue);
  sycl::free(scratchpad, queue);

  return 2.0*half_ldet.get();
}

template<class T>
void transpose(const int m, const int n, const T x, const T* A, const int ldA,
    T* B, const int ldB) {
  auto A1 = make_dpl_matrix_transpose(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(), A1.end(), B1.begin(), [=](T a) {
        return x*a; });
}

template<class T>
T trace(const int m, const int n, const T* A, const int ldA) {
  return sum(std::min(m, n), A, ldA + 1);
}

}