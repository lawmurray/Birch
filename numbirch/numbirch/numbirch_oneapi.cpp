/**
 * @file
 */
#include "numbirch/numbirch.hpp"

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <oneapi/dpl/ranges>
#include <oneapi/dpl/async>
#include <oneapi/mkl.hpp>
#include <CL/sycl.hpp>

using namespace cl;
using namespace oneapi;

namespace blas = mkl::blas::row_major;

/**
 * Thread-local SYCL queue.
 */
static thread_local auto queue =
    sycl::queue(sycl::device(sycl::host_selector()));

/**
 * Thread-local oneDPL policy.
 */
static thread_local auto policy =
    dpl::execution::make_device_policy(queue);

template<class T>
auto make_dpl_vector(T* x, const int n, const int incx) {
  return dpl::experimental::ranges::transform_view(
      dpl::experimental::ranges::iota_view(0, n), [=](int i) -> T& {
        return x[i*incx];
      });
}

template<class T>
auto make_dpl_matrix(T* A, const int m, const int n, const int ldA) {
  return dpl::experimental::ranges::transform_view(
      dpl::experimental::ranges::iota_view(0, m*n), [=](int i) -> T& {
        int c = i/m;
        int r = i%m;
        return A[r + c*ldA];
      });
}

template<class T>
auto make_dpl_matrix_transpose(T* A, const int m, const int n,
    const int ldA) {
  return dpl::experimental::ranges::transform_view(
      dpl::experimental::ranges::iota_view(0, m*n), [=](int i) -> T& {
        int r = i/n;
        int c = i%n;
        return A[r * c*ldA];
      });
}

template<class T>
auto make_dpl_matrix_lower(const T* A, const int m, const int n,
    const int ldA) {
  return dpl::experimental::ranges::transform_view(
      dpl::experimental::ranges::iota_view(0, m*n), [=](int i) -> T {
        int r = i/m;
        int c = i%m;
        return c <= r ? A[r*ldA + c] : 0.0;
      });
}

void numbirch::init() {
  //
}

void* numbirch::malloc(const size_t size) {
  return sycl::malloc_shared(size, queue);
}

void* numbirch::realloc(void* ptr, size_t oldsize, size_t newsize) {
  char* src = (char*)ptr;
  char* dst = sycl::malloc_shared<char>(newsize, queue);
  dpl::copy(policy, src, src + std::min(oldsize, newsize), dst);
  sycl::free(ptr, queue);
  return dst;
}

void numbirch::free(void* ptr) {
  sycl::free(ptr, queue);
}

void numbirch::copy(const int n, const double* x, const int incx, double* y,
    const int incy) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  dpl::copy(policy, x1.begin(), x1.end(), y1.begin());
}

void numbirch::copy(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::copy(policy, A1.begin(), A1.end(), B1.begin());
}

void numbirch::neg(const int n, const double* x, const int incx, double* y,
    const int incy) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  dpl::transform(policy, x1.begin(), x1.end(), y1.begin(),
      dpl::negate<double>());
}

void numbirch::neg(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::transform(policy, A1.begin(), A1.end(), B1.begin(),
      dpl::negate<double>());
}

void numbirch::add(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  auto z1 = make_dpl_vector(z, n, incz);
  dpl::transform(policy, x1.begin(),
      x1.end(), y1.begin(), z1.begin(), dpl::plus<double>());
}

void numbirch::add(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(policy, A1.begin(),
      A1.end(), B1.begin(), C1.begin(), dpl::plus<double>());
}

void numbirch::sub(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  auto z1 = make_dpl_vector(z, n, incz);
  dpl::transform(policy, x1.begin(),
      x1.end(), y1.begin(), z1.begin(), dpl::minus<double>());
}

void numbirch::sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(policy, A1.begin(),
      A1.end(), B1.begin(), C1.begin(), dpl::minus<double>());
}

void numbirch::hadamard(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  auto z1 = make_dpl_vector(z, n, incz);
  dpl::transform(policy, x1.begin(),
      x1.end(), y1.begin(), z1.begin(), dpl::multiplies<double>());
}

void numbirch::hadamard(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(policy, A1.begin(),
      A1.end(), B1.begin(), C1.begin(), dpl::multiplies<double>());
}

void numbirch::div(const int n, const double* x, const int incx,
    const double y, double* z, const int incz) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto z1 = make_dpl_vector(z, n, incz);
  dpl::transform(policy, x1.begin(), x1.end(), z1.begin(), [=](double x) {
        return x/y; });
}

void numbirch::div(const int m, const int n, const double* A, const int ldA,
    const double b, double* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(policy, A1.begin(), A1.end(), C1.begin(), [=](double a) {
        return a/b; });
}

void numbirch::mul(const int n, const double x, const double* y,
    const int incy, double* z, const int incz) {
  auto evt = blas::axpby(queue, n, x, y, incy, 0.0, z, incz);
  evt.wait();
}

void numbirch::mul(const int m, const int n, const double a, const double* B,
    const int ldB, double* C, const int ldC) {
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(policy, B1.begin(), B1.end(), C1.begin(), [=](double b) {
        return a*b; });
}

void numbirch::mul(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy) {
  auto evt = blas::gemv(queue, mkl::transpose::N, m, n, 1.0, A, ldA, x, incx,
      0.0, y, incy);
  evt.wait();
}

void numbirch::mul(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto evt = blas::gemm(queue, mkl::transpose::N, mkl::transpose::N, m, n, k,
      1.0, A, ldA, B, ldB, 0.0, C, ldC);
  evt.wait();
}

double numbirch::sum(const int n, const double* x, const int incx) {
  auto x1 = make_dpl_vector(x, n, incx);
  return dpl::reduce(policy, x1.begin(), x1.end());
}

double numbirch::sum(const int m, const int n, const double* A,
    const int ldA) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  return dpl::reduce(policy, A1.begin(), A1.end());
}

double numbirch::dot(const int n, const double* x, const int incx,
    const double* y, const int incy) {
  double z;
  auto evt = blas::dot(queue, n, x, incx, y, incy, &z);
  evt.wait();
  return z;
}

double numbirch::frobenius(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  return dpl::transform_reduce(policy, A1.begin(), A1.end(), B1.begin(), 0.0,
      dpl::plus<double>(), dpl::multiplies<double>());
}

void numbirch::inner(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy) {
  auto evt = blas::gemv(queue, mkl::transpose::T, m, n, 1.0, A, ldA, x, incx,
      0.0, y, incy);
  evt.wait();
}

void numbirch::inner(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto evt = blas::gemm(queue, mkl::transpose::T, mkl::transpose::N, m, n, k,
      1.0, A, ldA, B, ldB, 0.0, C, ldC);
  evt.wait();
}

void numbirch::outer(const int m, const int n, const double* x,
    const int incx, const double* y, const int incy, double* A,
    const int ldA) {
  auto evt = blas::gemm(queue, mkl::transpose::N, mkl::transpose::T, m, n, 1,
      1.0, x, incx, y, incy, 0.0, A, ldA);
  evt.wait();
}

void numbirch::outer(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto evt = blas::gemm(queue, mkl::transpose::N, mkl::transpose::T, m, n, k,
      1.0, A, ldA, B, ldB, 0.0, C, ldC);
  evt.wait();
}

void numbirch::solve(const int n, const double* A, const int ldA, double* x,
    const int incx, const double* y, const int incy) {
  solve(n, 1, A, ldA, x, incx, y, incy);
}

void numbirch::solve(const int m, const int n, const double* A, const int ldA,
    double* X, const int ldX, const double* Y, const int ldY) {
  auto scratchpad_size = mkl::lapack::getrf_scratchpad_size<double>(queue, m,
      m, ldA);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto ipiv = sycl::malloc_shared<int64_t>(std::max(1, m), queue);
  auto LU = sycl::malloc_shared<double>(std::max(1, m*m), queue);
  auto ldLU = m;

  /* solve via LU factorization with partial pivoting */
  auto evt1 = blas::copy_batch(queue, m, A, 1, ldA, LU, 1, ldLU, m);
  auto evt2 = mkl::lapack::getrf(queue, m, m, LU, ldLU, ipiv, scratchpad,
      scratchpad_size, {evt1});
  auto evt3 = blas::copy_batch(queue, m, Y, 1, ldY, X, 1, ldX, n);
  auto evt4 = mkl::lapack::getrs(queue, mkl::transpose::N, m, n, LU, ldLU,
      ipiv, X, ldX, scratchpad, scratchpad_size, {evt2, evt3});
  evt4.wait();

  sycl::free(LU, queue);
  sycl::free(ipiv, queue);
  sycl::free(scratchpad, queue);
}

void numbirch::cholsolve(const int n, const double* S, const int ldS,
    double* x, const int incx, const double* y, const int incy) {
  cholsolve(n, 1, S, ldS, x, incx, y, incy);
}

void numbirch::cholsolve(const int m, const int n, const double* S,
    const int ldS, double* X, const int ldX, const double* Y, const int ldY) {
  auto scratchpad_size = mkl::lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, m, ldS);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto LLT = sycl::malloc_shared<double>(m*m, queue);
  auto ldLLT = m;

  /* solve via Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, m, S, 1, ldS, LLT, 1, ldLLT, m);
  auto evt2 = mkl::lapack::potrf(queue, mkl::uplo::lower, m, LLT, ldLLT,
      scratchpad, scratchpad_size, {evt1});
  auto evt3 = blas::copy_batch(queue, m, Y, 1, ldY, X, 1, ldX, n);

  std::cerr << m << ' ' << n << ' ' << ldX << std::endl;

  auto evt4 = mkl::lapack::potrs(queue, mkl::uplo::lower, m, n, LLT, ldLLT, X,
      ldX, scratchpad, scratchpad_size, {evt2, evt3});
  evt4.wait();

  sycl::free(LLT, queue);
  sycl::free(scratchpad, queue);
}

void numbirch::inv(const int n, const double* A, const int ldA, double* B,
    const int ldB) {
  auto scratchpad_size1 = mkl::lapack::getrf_scratchpad_size<double>(queue, n,
      n, ldB);
  auto scratchpad_size2 = mkl::lapack::getri_scratchpad_size<double>(queue, n,
      ldB);
  auto scratchpad_size = std::max(scratchpad_size1, scratchpad_size2);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto ipiv = sycl::malloc_shared<int64_t>(std::max(1, n), queue);

  /* invert via LU factorization with partial pivoting */
  auto evt1 = blas::copy_batch(queue, n, A, 1, ldA, B, 1, ldB, n);
  auto evt2 = mkl::lapack::getrf(queue, n, n, B, ldB, ipiv, scratchpad,
      scratchpad_size, {evt1});
  auto evt3 = mkl::lapack::getri(queue, n, B, ldB, ipiv, scratchpad,
      scratchpad_size, {evt2});
  evt3.wait();

  sycl::free(ipiv, queue);
  sycl::free(scratchpad, queue);
}

void numbirch::cholinv(const int n, const double* S, const int ldS, double* B,
    const int ldB) {
  auto scratchpad_size1 = mkl::lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, n, ldB);
  auto scratchpad_size2 = mkl::lapack::potri_scratchpad_size<double>(queue,
      mkl::uplo::lower, n, ldB);
  auto scratchpad_size = std::max(scratchpad_size1, scratchpad_size2);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);

  /* invert via Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, B, 1, ldB, n);
  auto evt2 = mkl::lapack::potrf(queue, mkl::uplo::lower, n, B, ldB,
      scratchpad, scratchpad_size, {evt1});
  auto evt3 = mkl::lapack::potri(queue, mkl::uplo::lower, n, B, ldB,
      scratchpad, scratchpad_size, {evt2});
  evt3.wait();

  sycl::free(scratchpad, queue);
}

double numbirch::ldet(const int n, const double* A, const int ldA) {
  auto scratchpad_size = mkl::lapack::getrf_scratchpad_size<double>(queue, n,
      n, ldA);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto ipiv = sycl::malloc_shared<int64_t>(std::max(1, n), queue);
  auto LU = sycl::malloc_shared<double>(std::max(1, n*n), queue);
  auto ldLU = n;

  /* LU factorization with partial pivoting */
  auto evt1 = blas::copy_batch(queue, n, A, 1, ldA, LU, 1, ldLU, n);
  auto evt2 = mkl::lapack::getrf(queue, n, n, LU, ldLU, ipiv, scratchpad,
      scratchpad_size, {evt1});

  /* the LU factorization is with partial pivoting, which means $|A| = (-1)^p
   * |L||U|$, where $p$ is the number of row exchanges in `ipiv`; however,
   * we're taking the logarithm of its absolute value, so can ignore the first
   * term, and the second term is just 1 as $L$ has a unit diagonal; just need
   * $|U|$ here; the logarithm of its absolute value is just the sum of the
   * logarithms of the absolute values of elements on the main diagonal */
  auto d = make_dpl_vector(LU, n, ldLU + 1);  // diagonal of LU
  auto logabs = [](double x) { return std::log(std::abs(x)); };
  auto ldet = dpl::experimental::transform_reduce_async(policy, d.begin(),
      d.end(), 0.0, dpl::plus<double>(), logabs, evt2);
  ldet.wait();

  sycl::free(LU, queue);
  sycl::free(ipiv, queue);
  sycl::free(scratchpad, queue);

  return ldet.get();
}

double numbirch::lcholdet(const int n, const double* S, const int ldS) {
  auto scratchpad_size = mkl::lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto LLT = sycl::malloc_shared<double>(n*n, queue);
  auto ldLLT = n;

  /* Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, LLT, 1, ldLLT, n);
  auto evt2 = mkl::lapack::potrf(queue, mkl::uplo::lower, n, LLT, ldLLT,
      scratchpad, scratchpad_size, {evt1});

  /* log-determinant is twice the sum of logarithms of elements on the main
   * diagonal, all of which should be positive; the 2.0 is multiplied in by
   * the return statement below */
  auto d = make_dpl_vector(LLT, n, ldLLT + 1);  // diagonal of LLT
  auto log = [](double x) { return std::log(x); };
  auto half_ldet = dpl::experimental::transform_reduce_async(policy,
      d.begin(), d.end(), 0.0, dpl::plus<double>(), log, evt2);
  half_ldet.wait();

  sycl::free(LLT, queue);
  sycl::free(scratchpad, queue);

  return 2.0*half_ldet.get();
}

void numbirch::chol(const int n, const double* S, const int ldS, double* L,
    const int ldL) {
  auto scratchpad_size = mkl::lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, n, ldL);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);

  /* copy lower triangle, and zero upper triangle */
  auto S1 = make_dpl_matrix_lower(S, n, n, ldS);
  auto L1 = make_dpl_matrix(L, n, n, ldL);
  //auto evt1 = dpl::experimental::copy_async(policy, S1.begin(), S1.end(),
  //    L1.begin());
  // ^ compile errors as of Intel oneDPL 2021.4.0
  dpl::copy(policy, S1.begin(), S1.end(), L1.begin());
  auto evt2 = mkl::lapack::potrf(queue, mkl::uplo::lower, n, L, ldL,
      scratchpad, scratchpad_size);
  evt2.wait();

  sycl::free(scratchpad, queue);
}

void numbirch::transpose(const int m, const int n, const double x,
    const double* A, const int ldA, double* B, const int ldB) {
  auto A1 = make_dpl_matrix_transpose(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::transform(policy, A1.begin(), A1.end(), B1.begin(), [=](double a) {
        return x*a; });
}

double numbirch::trace(const int m, const int n, const double* A,
    const int ldA) {
  return sum(m, n, A, ldA + 1);
}
