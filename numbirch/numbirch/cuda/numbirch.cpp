/**
 * @file
 * 
 * CUDA implementation of numerical functions.
 */
#include "numbirch/numbirch.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/thrust.hpp"

#include <jemalloc/jemalloc.h>
#include <omp.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

static thread_local cudaStream_t stream = cudaStreamPerThread;
static thread_local cublasHandle_t cublasHandle;
static thread_local cusolverDnHandle_t cusolverDnHandle;
static thread_local cusolverDnParams_t cusolverDnParams;
static thread_local auto policy = thrust::cuda::par.on(stream);

static double* one = nullptr;
static double* zero = nullptr;
static int* info = nullptr;

static thread_local unsigned host_arena = 0;
static thread_local unsigned host_tcache = 0;
static thread_local int host_flags = 0;

static thread_local unsigned device_arena = 0;
static thread_local unsigned device_tcache = 0;
static thread_local int device_flags = 0;

void* extent_alloc(extent_hooks_t *extent_hooks, void *new_addr, size_t size,
    size_t alignment, bool *zero, bool *commit, unsigned arena_ind) {
  if (!new_addr) {
    CUDA_CHECK(cudaMallocManaged(&new_addr, size));
  }
  if (*zero) {
    CUDA_CHECK(cudaMemset(new_addr, 0, size));
  }
  return new_addr;
}

bool extent_dalloc(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFree(addr));
  return false;
}

void extent_destroy(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFree(addr));
}

static extent_hooks_t hooks = {
  extent_alloc,
  extent_dalloc,
  extent_destroy,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr,
  nullptr
};

unsigned make_arena() {
  unsigned arena = 0;
  void* hooks1 = &hooks;
  size_t size = sizeof(arena);
  [[maybe_unused]] int ret = mallctl("arenas.create", &arena, &size, &hooks1,
      sizeof(hooks1));
  assert(ret == 0);
  return arena;
}

unsigned make_tcache() {
  unsigned tcache = 0;
  size_t size = sizeof(tcache);
  [[maybe_unused]] int ret = mallctl("tcache.create", &tcache, &size, nullptr,
      0);
  assert(ret == 0);
  return tcache;
}

void* device_malloc(const size_t size) {
  assert(device_arena > 0);
  return size == 0 ? nullptr : mallocx(size, device_flags);
}

void device_free(void* ptr) {
  assert(device_arena > 0);
  if (ptr) {
    dallocx(ptr, device_flags);
  }
}

void numbirch::init() {
  double one1 = 1.0;
  double zero1 = 0.0;

  CUDA_CHECK(cudaMalloc(&one, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&zero, sizeof(double)));  
  CUBLAS_CHECK(cublasSetVector(1, sizeof(double), &one1, 1, one, 1));
  CUBLAS_CHECK(cublasSetVector(1, sizeof(double), &zero1, 1, zero, 1));

  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
    CUBLAS_CHECK(cublasSetPointerMode(cublasHandle,
        CUBLAS_POINTER_MODE_DEVICE));

    CUSOLVER_CHECK(cusolverDnCreate(&cusolverDnHandle));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverDnHandle, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverDnParams));

    host_arena = make_arena();
    host_tcache = make_tcache();
    host_flags = MALLOCX_ARENA(host_arena)|MALLOCX_TCACHE(host_tcache);

    device_arena = make_arena();
    device_tcache = make_tcache();
    device_flags = MALLOCX_ARENA(device_arena)|MALLOCX_TCACHE(device_tcache);
  }
}

void numbirch::term() {
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverDnParams));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverDnHandle));
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
  }
  CUDA_CHECK(cudaFree(zero));
  CUDA_CHECK(cudaFree(one));
  CUDA_CHECK(cudaDeviceSynchronize());
}

void* numbirch::malloc(const size_t size) {
  assert(host_arena > 0);
  return size == 0 ? nullptr : mallocx(size, host_flags);
}

void* numbirch::realloc(void* ptr, const size_t size) {
  assert(host_arena > 0);
  if (size > 0) {
    return rallocx(ptr, size, host_flags);
  } else {
    free(ptr);
    return nullptr;
  }
}

void numbirch::free(void* ptr) {
  assert(host_arena > 0);
  if (ptr) {
    dallocx(ptr, host_flags);
  }
}

void numbirch::memcpy(void* dst, const size_t dpitch, const void* src,
    const size_t spitch, const size_t width, const size_t height) {
  CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
      cudaMemcpyDefault, stream));
}

void numbirch::wait() {
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void numbirch::neg(const int n, const double* x, const int incx, double* y,
    const int incy) {
  auto x1 = make_thrust_vector(x, n, incx);
  auto y1 = make_thrust_vector(y, n, incy);
  thrust::async::transform(policy, x1.begin(), x1.end(), y1.begin(),
      thrust::negate<double>());
}

void numbirch::neg(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB) {
  auto A1 = make_thrust_matrix(A, m, n, ldA);
  auto B1 = make_thrust_matrix(B, m, n, ldB);
  thrust::transform(policy, A1.begin(), A1.end(), B1.begin(),
      thrust::negate<double>());
}

void numbirch::add(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_thrust_vector(x, n, incx);
  auto y1 = make_thrust_vector(y, n, incy);
  auto z1 = make_thrust_vector(z, n, incz);
  thrust::transform(policy, x1.begin(), x1.end(), y1.begin(),
      z1.begin(), thrust::plus<double>());
}

void numbirch::add(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_thrust_matrix(A, m, n, ldA);
  auto B1 = make_thrust_matrix(B, m, n, ldB);
  auto C1 = make_thrust_matrix(C, m, n, ldC);
  thrust::transform(policy, A1.begin(), A1.end(), B1.begin(),
      C1.begin(), thrust::plus<double>());
}

void numbirch::sub(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_thrust_vector(x, n, incx);
  auto y1 = make_thrust_vector(y, n, incy);
  auto z1 = make_thrust_vector(z, n, incz);
  thrust::transform(policy, x1.begin(), x1.end(), y1.begin(), z1.begin(),
      thrust::minus<double>());
}

void numbirch::sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_thrust_matrix(A, m, n, ldA);
  auto B1 = make_thrust_matrix(B, m, n, ldB);
  auto C1 = make_thrust_matrix(C, m, n, ldC);
  thrust::transform(policy, A1.begin(), A1.end(), B1.begin(), C1.begin(),
      thrust::minus<double>());
}

void numbirch::hadamard(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  auto x1 = make_thrust_vector(x, n, incx);
  auto y1 = make_thrust_vector(y, n, incy);
  auto z1 = make_thrust_vector(z, n, incz);
  thrust::transform(policy, x1.begin(), x1.end(), y1.begin(), z1.begin(),
      thrust::multiplies<double>());
}

void numbirch::hadamard(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  auto A1 = make_thrust_matrix(A, m, n, ldA);
  auto B1 = make_thrust_matrix(B, m, n, ldB);
  auto C1 = make_thrust_matrix(C, m, n, ldC);
  thrust::transform(policy, A1.begin(), A1.end(), B1.begin(), C1.begin(),
      thrust::multiplies<double>());
}

void numbirch::div(const int n, const double* x, const int incx,
    const double y, double* z, const int incz) {
  auto x1 = make_thrust_vector(x, n, incx);
  auto z1 = make_thrust_vector(z, n, incz);
  thrust::async::transform(policy, x1.begin(), x1.end(), z1.begin(),
      ScalarDivideFunctor(y));
}

void numbirch::div(const int m, const int n, const double* A, const int ldA,
    const double b, double* C, const int ldC) {
  auto A1 = make_thrust_matrix(A, m, n, ldA);
  auto C1 = make_thrust_matrix(C, m, n, ldC);
  thrust::async::transform(policy, A1.begin(), A1.end(), C1.begin(),
      ScalarDivideFunctor(b));
}

void numbirch::mul(const int n, const double x, const double* y,
    const int incy, double* z, const int incz) {
  auto y1 = make_thrust_vector(y, n, incy);
  auto z1 = make_thrust_vector(z, n, incz);
  thrust::async::transform(policy, y1.begin(), y1.end(), z1.begin(),
      ScalarMultiplyFunctor(x));
}

void numbirch::mul(const int m, const int n, const double a, const double* B,
    const int ldB, double* C, const int ldC) {
  auto B1 = make_thrust_matrix(B, m, n, ldB);
  auto C1 = make_thrust_matrix(C, m, n, ldC);
  thrust::transform(policy, B1.begin(), B1.end(), C1.begin(),
      ScalarMultiplyFunctor(a));
}

void numbirch::mul(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy) {
  // CUDA_CHECK(cudaMemPrefetchAsync(A, n*ldA*sizeof(double), device, stream));
  // CUDA_CHECK(cudaMemPrefetchAsync(x, n*incx*sizeof(double), device, stream));
  // CUDA_CHECK(cudaMemPrefetchAsync(y, m*incy*sizeof(double), device, stream));

  CUBLAS_CHECK(cublasDgemv(cublasHandle, CUBLAS_OP_N, m, n, one, A, ldA, x, 
      incx, zero, y, incy));
}

void numbirch::mul(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  // CUDA_CHECK(cudaMemPrefetchAsync(A, k*ldA*sizeof(double), device, stream));
  // CUDA_CHECK(cudaMemPrefetchAsync(B, n*ldB*sizeof(double), device, stream));
  // CUDA_CHECK(cudaMemPrefetchAsync(C, n*ldC*sizeof(double), device, stream));

  CUBLAS_CHECK(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
      one, A, ldA, B, ldB, zero, C, ldC));
}

void numbirch::cholmul(const int n, const double* S, const int ldS,
    const double* x, const int incx, double* y, const int incy) {
  auto ldL = n;
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, n*n));

  // CUDA_CHECK(cudaMemPrefetchAsync(S, n*ldS*sizeof(double), device, stream));
  // CUDA_CHECK(cudaMemPrefetchAsync(x, n*incx*sizeof(double), device, stream));
  // CUDA_CHECK(cudaMemPrefetchAsync(y, n*incy*sizeof(double), device, stream));

  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L, ldL,
      CUDA_R_64F, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = device_malloc(bufferOnHostBytes);

  /* Cholesky factorization */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), S, ldS*sizeof(double),
      n*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L, ldL, CUDA_R_64F,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));

  /* multiplication */
  CUBLAS_CHECK(cublasDcopy(cublasHandle, n, x, incx, y, incy));
  CUBLAS_CHECK(cublasDtrmv(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
      CUBLAS_DIAG_NON_UNIT, n, L, ldL, y, incy));

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

void numbirch::cholmul(const int m, const int n, const double* S,
    const int ldS, const double* B, const int ldB, double* C, const int ldC) {
  // CUDA_CHECK(cudaMemPrefetchAsync(S, m*ldS*sizeof(double), device, stream));
  // CUDA_CHECK(cudaMemPrefetchAsync(B, n*ldB*sizeof(double), device, stream));
  // CUDA_CHECK(cudaMemPrefetchAsync(C, n*ldC*sizeof(double), device, stream));

  auto ldL = m;
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, m*m));
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, m, CUDA_R_64F, L, ldL,
      CUDA_R_64F, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = device_malloc(bufferOnHostBytes);

  /* Cholesky factorization */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), S, ldS*sizeof(double),
      m*sizeof(double), m, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, m, CUDA_R_64F, L, ldL, CUDA_R_64F,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));

  /* multiplication */
  CUBLAS_CHECK(cublasDtrmm(cublasHandle, CUBLAS_SIDE_LEFT,
      CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, one, L,
      ldL, B, ldB, C, ldC));

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

double numbirch::sum(const int n, const double* x, const int incx) {
  auto x1 = make_thrust_vector(x, n, incx);
  return thrust::reduce(policy, x1.begin(), x1.end());
}

double numbirch::sum(const int m, const int n, const double* A,
    const int ldA) {
  auto A1 = make_thrust_matrix(A, m, n, ldA);
  return thrust::reduce(policy, A1.begin(), A1.end());
}

double numbirch::dot(const int n, const double* x, const int incx,
    const double* y, const int incy) {
  double* z = (double*)malloc(sizeof(double));
  CUBLAS_CHECK(cublasDdot(cublasHandle, n, x, incx, y, incy, z));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  double result = *z;
  free(z);
  return result;
}

double numbirch::frobenius(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB) {
  auto A1 = make_thrust_matrix(A, m, n, ldA);
  auto B1 = make_thrust_matrix(B, m, n, ldB);
  return thrust::inner_product(policy, A1.begin(), A1.end(), B1.begin(), 0.0);
}

void numbirch::inner(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy) {
  CUBLAS_CHECK(cublasDgemv(cublasHandle, CUBLAS_OP_T, n, m, one, A, ldA, x,
      incx, zero, y, incy));
}

void numbirch::inner(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  CUBLAS_CHECK(cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
      one, A, ldA, B, ldB, zero, C, ldC));
}

void numbirch::outer(const int m, const int n, const double* x,
    const int incx, const double* y, const int incy, double* A,
    const int ldA) {
  /* here, the two vectors are interpreted as single-row matrices, so that the
   * stride between elements becomes the stride between columns; to create the
   * outer product, the first matrix is transposed to a single-column matrix,
   * while the second is not */
  CUBLAS_CHECK(cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, 1,
      one, x, incx, y, incy, zero, A, ldA));
}

void numbirch::outer(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC) {
  CUBLAS_CHECK(cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k,
      one, A, ldA, B, ldB, zero, C, ldC));
}

void numbirch::cholouter(const int m, const int n, const double* A,
    const int ldA, const double* S, const int ldS, double* C, const int ldC) {
  auto ldL = n;
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, n*n));
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L, ldL,
      CUDA_R_64F, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = device_malloc(bufferOnHostBytes);

  /* Cholesky factorization */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), S, ldS*sizeof(double),
      n*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L, ldL, CUDA_R_64F,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));

  /* multiplication */
  CUBLAS_CHECK(cublasDtrmm(cublasHandle, CUBLAS_SIDE_RIGHT,
      CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, n, one, L,
      ldL, A, ldA, C, ldC));

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

void numbirch::solve(const int n, const double* A, const int ldA, double* x,
    const int incx, const double* y, const int incy) {
  auto ldL = n;
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*std::max(1, n));
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, n*n));
  auto x1 = x;
  if (incx > 1) {
    x1 = (double*)device_malloc(sizeof(double)*n);
  }
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, n, n, CUDA_R_64F, L, ldL, CUDA_R_64F,
      &bufferOnDeviceBytes, &bufferOnHostBytes));
  void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = device_malloc(bufferOnHostBytes);

  /* solve via L factorization with partial pivoting */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), A, ldA*sizeof(double),
    n*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n, n,
      CUDA_R_64F, L, ldL, ipiv, CUDA_R_64F, bufferOnDevice,
      bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes, info));
  CUBLAS_CHECK(cublasDcopy(cublasHandle, n, y, incy, x1, 1));
  CUSOLVER_CHECK(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_OP_N, n, 1, CUDA_R_64F, L, ldL, ipiv, CUDA_R_64F, x1, n,
      info));
  if (incx > 1) {
    CUBLAS_CHECK(cublasDcopy(cublasHandle, n, x1, 1, x, incx));
  }

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  if (incx > 1) {
    device_free(x1);
  }
  device_free(L);
  device_free(ipiv);
}

void numbirch::solve(const int m, const int n, const double* A, const int ldA,
    double* X, const int ldX, const double* Y, const int ldY) {
  auto ldL = m;
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*std::min(m, n));
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, m*m));
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, m, m, CUDA_R_64F, L, ldL, CUDA_R_64F,
      &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = device_malloc(bufferOnHostBytes);

  /* solve via L factorization with partial pivoting */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), A, ldA*sizeof(double),
    n*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n, n,
      CUDA_R_64F, L, ldL, ipiv, CUDA_R_64F, bufferOnDevice,
      bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes, info));
  CUDA_CHECK(cudaMemcpy2DAsync(X, ldX*sizeof(double), Y, ldY*sizeof(double),
    m*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_OP_N, m, n, CUDA_R_64F, L, ldL, ipiv, CUDA_R_64F, X, ldX,
      info));

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
  device_free(ipiv);
}

void numbirch::cholsolve(const int n, const double* S, const int ldS,
    double* x, const int incx, const double* y, const int incy) {
  auto ldL = n;
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, n*n));
  double* x1 = x;
  if (incx > 1) {
    x1 = (double*)device_malloc(sizeof(double)*n);
  }
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L, ldL,
      CUDA_R_64F, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = device_malloc(bufferOnHostBytes);

  /* solve via Cholesky factorization */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), S,
      ldS*sizeof(double), n*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L, ldL, CUDA_R_64F,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));
  CUBLAS_CHECK(cublasDcopy(cublasHandle, n, y, incy, x1, 1));
  CUSOLVER_CHECK(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, 1, CUDA_R_64F, L, ldL, CUDA_R_64F, x1,
      n, info));
  if (incx > 1) {
    CUBLAS_CHECK(cublasDcopy(cublasHandle, n, x1, 1, x, incx));
  }

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  if (incx > 1) {
    device_free(x1);
  }
  device_free(L);
}

void numbirch::cholsolve(const int m, const int n, const double* S,
    const int ldS, double* X, const int ldX, const double* Y, const int ldY) {
  auto ldL = m;
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, m*m));
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, m, CUDA_R_64F, L, ldL,
      CUDA_R_64F, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = device_malloc(bufferOnHostBytes);

  /* solve via Cholesky factorization */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), S,
      ldS*sizeof(double), m*sizeof(double), m, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, m, CUDA_R_64F, L, ldL, CUDA_R_64F,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));
  CUDA_CHECK(cudaMemcpy2DAsync(X, ldX*sizeof(double), Y, ldY*sizeof(double),
    m*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, m, n, CUDA_R_64F, L, ldL, CUDA_R_64F, X,
      ldX, info));

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

void numbirch::inv(const int n, const double* A, const int ldA, double* B,
    const int ldB) {
  auto ldL = n;
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*n);
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, n*n));
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, n, n, CUDA_R_64F, L, ldL, CUDA_R_64F,
      &bufferOnDeviceBytes, &bufferOnHostBytes));
  void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = device_malloc(bufferOnHostBytes);

  /* L factorization with partial pivoting */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), A, ldA*sizeof(double),
      n*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n, n,
      CUDA_R_64F, L, ldL, ipiv, CUDA_R_64F, bufferOnDevice,
      bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes, info));

  /* write identity matrix into B */
  CUDA_CHECK(cudaMemset2DAsync(B, ldB*sizeof(double), 0, n*sizeof(double),
      n));
  CUBLAS_CHECK(cublasDcopy(cublasHandle, n, one, 0, B, ldB + 1));

  /* solve */
  CUSOLVER_CHECK(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_OP_N, n, n, CUDA_R_64F, L, ldL, ipiv, CUDA_R_64F, B, ldB, info));

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
  device_free(ipiv);
}

void numbirch::cholinv(const int n, const double* S, const int ldS, double* B,
    const int ldB) {
  auto ldL = n;
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, n*n));
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L, ldL,
      CUDA_R_64F, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = device_malloc(bufferOnHostBytes);

  /* Cholesky factorization */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), S,
      ldS*sizeof(double), n*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L, ldL, CUDA_R_64F,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));

  /* write identity matrix into B */
  CUDA_CHECK(cudaMemset2DAsync(B, ldB*sizeof(double), 0, n*sizeof(double),
      n));
  CUBLAS_CHECK(cublasDcopy(cublasHandle, n, one, 0, B, ldB + 1));

  /* solve */
  CUSOLVER_CHECK(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, n, CUDA_R_64F, L, ldL, CUDA_R_64F, B,
      ldB, info));

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

double numbirch::ldet(const int n, const double* A, const int ldA) {
  auto ldL = n;
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*n);
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, n*n));
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, n, n, CUDA_R_64F, L, ldL, CUDA_R_64F,
      &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = device_malloc(bufferOnHostBytes);

  /* L factorization with partial pivoting */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), A, ldA*sizeof(double),
      n*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n, n,
      CUDA_R_64F, L, ldL, ipiv, CUDA_R_64F, bufferOnDevice,
      bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes, info));

  /* the L factorization is with partial pivoting, which means $|A| = (-1)^p
   * |L||U|$, where $p$ is the number of row exchanges in `ipiv`; however,
   * we're taking the logarithm of its absolute value, so can ignore the first
   * term, and the second term is just 1 as $L$ has a unit diagonal; just need
   * $|U|$ here; the logarithm of its absolute value is just the sum of the
   * logarithms of the absolute values of elements on the main diagonal */
  auto d = make_thrust_vector(L, n, ldL + 1);  // diagonal of L
  double ldet = thrust::transform_reduce(policy, d.begin(), d.end(),
      LogAbsFunctor<double>(), 0.0, thrust::plus<double>());

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
  device_free(ipiv);

  return ldet;
}

double numbirch::lcholdet(const int n, const double* S, const int ldS) {
  auto ldL = n;
  auto L = (double*)device_malloc(sizeof(double)*std::max(1, n*n));
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L, ldL,
      CUDA_R_64F, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = device_malloc(bufferOnHostBytes);

  /* solve via Cholesky factorization */
  CUDA_CHECK(cudaMemcpy2DAsync(L, ldL*sizeof(double), S,
      ldS*sizeof(double), n*sizeof(double), n, cudaMemcpyDefault, stream));
  CUSOLVER_CHECK(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, L, ldL, CUDA_R_64F,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));

  /* log-determinant is twice the sum of logarithms of elements on the main
   * diagonal, all of which should be positive */
  auto d = make_thrust_vector(L, n, ldL + 1);  // diagonal of L
  double ldet = 2.0*thrust::transform_reduce(policy, d.begin(), d.end(),
      LogFunctor<double>(), 0.0, thrust::plus<double>());

  device_free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);

  return ldet;
}

void numbirch::transpose(const int m, const int n, const double x,
    const double* A, const int ldA, double* B, const int ldB) {
  auto A1 = make_thrust_matrix_transpose(A, m, n, ldA);
  auto B1 = make_thrust_matrix(B, m, n, ldB);
  thrust::async::transform(policy, A1.begin(), A1.end(), B1.begin(),
      ScalarMultiplyFunctor(x));
}

double numbirch::trace(const int m, const int n, const double* A,
    const int ldA) {
  return sum(std::min(m, n), A, ldA + 1);
}
