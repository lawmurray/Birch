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
#include <omp.h>

using namespace cl;
using namespace oneapi;

namespace blas = mkl::blas::column_major;
namespace lapack = mkl::lapack;

static auto device = sycl::device(sycl::gpu_selector());
static auto context = sycl::context(device);
static auto queue = sycl::queue(context, device);
static auto policy = dpl::execution::make_device_policy(queue);

template<class T>
static auto make_dpl_vector(T* x, const int n, const int incx) {
  return dpl::experimental::ranges::transform_view(
      dpl::experimental::ranges::iota_view(0, n), [=](int i) -> T& {
        return x[i*incx];
      });
}

template<class T>
static auto make_dpl_matrix(T* A, const int m, const int n, const int ldA) {
  return dpl::experimental::ranges::transform_view(
      dpl::experimental::ranges::iota_view(0, m*n), [=](int i) -> T& {
        int c = i/m;
        int r = i - c*m;
        return A[r + c*ldA];
      });
}

template<class T>
static auto make_dpl_matrix_transpose(T* A, const int m, const int n,
    const int ldA) {
  return dpl::experimental::ranges::transform_view(
      dpl::experimental::ranges::iota_view(0, m*n), [=](int i) -> T& {
        int c = i/m;
        int r = i - c*m;
        return A[c + r*ldA];
      });
}

template<class T>
static auto make_dpl_matrix_lower(const T* A, const int m, const int n,
    const int ldA) {
  return dpl::experimental::ranges::transform_view(
      dpl::experimental::ranges::iota_view(0, m*n), [=](int i) -> T {
        int c = i/m;
        int r = i - c*m;
        return (c <= r) ? A[r + c*ldA] : 0.0;
      });
}

template<class T>
static auto make_dpl_matrix_symmetric(const T* A, const int m, const int n,
    const int ldA) {
  return dpl::experimental::ranges::transform_view(
      dpl::experimental::ranges::iota_view(0, m*n), [=](int i) -> T {
        int c = i/m;
        int r = i - c*m;
        return A[(c <= r) ? (r + c*ldA) : (c + r*ldA)];
      });
}

void numbirch::init() {
  //
}

void* numbirch::malloc(const size_t size) {
  void* ptr = sycl::malloc_shared(size, queue);
  queue.wait();
  return ptr;
}

void* numbirch::realloc(void* ptr, size_t oldsize, size_t newsize) {
  void* dst = sycl::malloc_shared(newsize, queue);
  queue.memcpy(dst, ptr, std::min(oldsize, newsize));
  sycl::free(ptr, queue);
  queue.wait();
  return dst;
}

void numbirch::free(void* ptr) {
  sycl::free(ptr, queue);
  queue.wait();
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
  dpl::transform(policy, x1.begin(), x1.end(), y1.begin(), z1.begin(),
      dpl::plus<double>());
}

void numbirch::add(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(policy, A1.begin(), A1.end(), B1.begin(), C1.begin(),
      dpl::plus<double>());
}

void numbirch::sub(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  auto z1 = make_dpl_vector(z, n, incz);
  dpl::transform(policy, x1.begin(), x1.end(), y1.begin(), z1.begin(),
      dpl::minus<double>());
}

void numbirch::sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(policy, A1.begin(), A1.end(), B1.begin(), C1.begin(),
      dpl::minus<double>());
}

void numbirch::hadamard(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_dpl_vector(x, n, incx);
  auto y1 = make_dpl_vector(y, n, incy);
  auto z1 = make_dpl_vector(z, n, incz);
  dpl::transform(policy, x1.begin(), x1.end(), y1.begin(), z1.begin(),
      dpl::multiplies<double>());
}

void numbirch::hadamard(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  dpl::transform(policy, A1.begin(), A1.end(), B1.begin(), C1.begin(),
      dpl::multiplies<double>());
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

void numbirch::cholmul(const int n, const double* S, const int ldS,
    const double* x, const int incx, double* y, const int incy) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<double>(n*n, queue);
  auto ldL = n;

  /* Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, n, L, ldL, scratchpad,
      scratchpad_size, {evt1});

  /* multiply */
  auto evt3 = blas::copy(queue, n, x, incx, y, incy);
  auto evt4 = blas::trmv(queue, mkl::uplo::lower, mkl::transpose::N,
      mkl::diag::N, n, L, ldL, y, incy, {evt2, evt3});
  evt4.wait();

  sycl::free(L, queue);
  sycl::free(scratchpad, queue);
}

void numbirch::cholmul(const int m, const int n, const double* S,
    const int ldS, const double* B, const int ldB, double* C, const int ldC) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, m, ldS);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<double>(m*m, queue);
  auto ldL = m;

  /* Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, m, S, 1, ldS, L, 1, ldL, m);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, m, L, ldL, scratchpad,
      scratchpad_size, {evt1});

  /* multiply */
  auto evt3 = blas::copy_batch(queue, m, B, 1, ldB, C, 1, ldC, n);
  auto evt4 = blas::trmm(queue, mkl::side::left, mkl::uplo::lower,
      mkl::transpose::N, mkl::diag::N, m, n, 1.0, L, ldL, C, ldC,
      {evt2, evt3});
  evt4.wait();

  sycl::free(L, queue);
  sycl::free(scratchpad, queue);
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
  auto z = sycl::malloc_shared<double>(1, queue);
  auto evt = blas::dot(queue, n, x, incx, y, incy, z);
  evt.wait();
  auto res = z[0];
  sycl::free(z, queue);
  return res;
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
  auto evt = blas::gemv(queue, mkl::transpose::T, n, m, 1.0, A, ldA, x, incx,
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
  /* here, the two vectors are interpreted as single-row matrices, so that the
   * stride between elements becomes the stride between columns; to create the
   * outer product, the first matrix is transposed to a single-column matrix,
   * while the second is not */
  auto evt = blas::gemm(queue, mkl::transpose::T, mkl::transpose::N, m, n, 1,
      1.0, x, incx, y, incy, 0.0, A, ldA);
  evt.wait();
}

void numbirch::outer(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto evt = blas::gemm(queue, mkl::transpose::N, mkl::transpose::T, m, n, k,
      1.0, A, ldA, B, ldB, 0.0, C, ldC);
  evt.wait();
}

void numbirch::cholouter(const int m, const int n, const double* A,
    const int ldA, const double* S, const int ldS, double* C, const int ldC) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<double>(n*n, queue);
  auto ldL = n;

  /* Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, n, L, ldL, scratchpad,
      scratchpad_size, {evt1});

  /* multiply */
  auto evt3 = blas::copy_batch(queue, m, A, 1, ldA, C, 1, ldC, n);
  auto evt4 = blas::trmm(queue, mkl::side::right, mkl::uplo::lower,
      mkl::transpose::T, mkl::diag::N, m, n, 1.0, L, ldL, C, ldC,
      {evt2, evt3});
  evt4.wait();

  sycl::free(L, queue);
  sycl::free(scratchpad, queue);
}

void numbirch::solve(const int n, const double* A, const int ldA, double* x,
    const int incx, const double* y, const int incy) {
  auto scratchpad_size = lapack::getrf_scratchpad_size<double>(queue, n,
      n, ldA);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto ipiv = sycl::malloc_shared<int64_t>(std::max(1, n), queue);
  auto L = sycl::malloc_shared<double>(std::max(1, n*n), queue);
  auto ldL = n;
  double* x1 = x;
  if (incx > 1) {
    x1 = sycl::malloc_shared<double>(n, queue);
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

void numbirch::solve(const int m, const int n, const double* A, const int ldA,
    double* X, const int ldX, const double* Y, const int ldY) {
  auto scratchpad_size = lapack::getrf_scratchpad_size<double>(queue, m,
      m, ldA);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto ipiv = sycl::malloc_shared<int64_t>(std::max(1, m), queue);
  auto L = sycl::malloc_shared<double>(std::max(1, m*m), queue);
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

void numbirch::cholsolve(const int n, const double* S, const int ldS,
    double* x, const int incx, const double* y, const int incy) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<double>(n*n, queue);
  auto ldL = n;
  double* x1 = x;
  if (incx > 1) {
    x1 = sycl::malloc_shared<double>(n, queue);
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

void numbirch::cholsolve(const int m, const int n, const double* S,
    const int ldS, double* X, const int ldX, const double* Y, const int ldY) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, m, ldS);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<double>(m*m, queue);
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

void numbirch::inv(const int n, const double* A, const int ldA, double* B,
    const int ldB) {
  auto scratchpad_size1 = lapack::getrf_scratchpad_size<double>(queue, n,
      n, ldB);
  auto scratchpad_size2 = lapack::getri_scratchpad_size<double>(queue, n,
      ldB);
  auto scratchpad_size = std::max(scratchpad_size1, scratchpad_size2);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
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

void numbirch::cholinv(const int n, const double* S, const int ldS, double* B,
    const int ldB) {
  auto scratchpad_size1 = lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, n, ldB);
  auto scratchpad_size2 = lapack::potri_scratchpad_size<double>(queue,
      mkl::uplo::lower, n, ldB);
  auto scratchpad_size = std::max(scratchpad_size1, scratchpad_size2);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto A = sycl::malloc_shared<double>(n*n, queue);
  auto ldA = n;

  /* invert via Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, A, 1, ldA, n);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, n, A, ldA, scratchpad,
      scratchpad_size, {evt1});
  auto evt3 = lapack::potri(queue, mkl::uplo::lower, n, A, ldA, scratchpad,
      scratchpad_size, {evt2});

  /* potri only modifies the lower triangle of A, whereas caller expects the
   * whole symmetric matrix (at least for now); copy that into B */
  auto A1 = make_dpl_matrix_symmetric(A, n, n, ldA);
  auto B1 = make_dpl_matrix(B, n, n, ldB);
  // auto evt4 = dpl::experimental::copy_async(policy, A1.begin(), A1.end(),
  //    B1.begin(), evt3);
  // evt4.wait();
  // ^ compile errors as of Intel oneDPL 2021.4.0
  evt3.wait();
  dpl::copy(policy, A1.begin(), A1.end(), B1.begin());

  sycl::free(A, queue);
  sycl::free(scratchpad, queue);
}

double numbirch::ldet(const int n, const double* A, const int ldA) {
  auto scratchpad_size = lapack::getrf_scratchpad_size<double>(queue, n,
      n, ldA);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto ipiv = sycl::malloc_shared<int64_t>(std::max(1, n), queue);
  auto L = sycl::malloc_shared<double>(std::max(1, n*n), queue);
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
  auto logabs = [](double x) { return std::log(std::abs(x)); };
  auto ldet = dpl::experimental::transform_reduce_async(policy, d.begin(),
      d.end(), 0.0, dpl::plus<double>(), logabs, evt2);
  ldet.wait();

  sycl::free(L, queue);
  sycl::free(ipiv, queue);
  sycl::free(scratchpad, queue);

  return ldet.get();
}

double numbirch::lcholdet(const int n, const double* S, const int ldS) {
  auto scratchpad_size = lapack::potrf_scratchpad_size<double>(queue,
      mkl::uplo::lower, n, ldS);
  auto scratchpad = sycl::malloc_shared<double>(scratchpad_size, queue);
  auto L = sycl::malloc_shared<double>(n*n, queue);
  auto ldL = n;

  /* Cholesky factorization */
  auto evt1 = blas::copy_batch(queue, n, S, 1, ldS, L, 1, ldL, n);
  auto evt2 = lapack::potrf(queue, mkl::uplo::lower, n, L, ldL,
      scratchpad, scratchpad_size, {evt1});

  /* log-determinant is twice the sum of logarithms of elements on the main
   * diagonal, all of which should be positive; the 2.0 is multiplied in by
   * the return statement below */
  auto d = make_dpl_vector(L, n, ldL + 1);  // diagonal of L
  auto log = [](double x) { return std::log(x); };
  auto half_ldet = dpl::experimental::transform_reduce_async(policy,
      d.begin(), d.end(), 0.0, dpl::plus<double>(), log, evt2);
  half_ldet.wait();

  sycl::free(L, queue);
  sycl::free(scratchpad, queue);

  return 2.0*half_ldet.get();
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
  return sum(std::min(m, n), A, ldA + 1);
}
