/**
 * @file
 * 
 * Generic implementation of interface for CUDA.
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/cub.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"

namespace numbirch {
/*
 * Tile size (number of rows, number of columns) for transpose().
 */
static const int TRANSPOSE_SIZE = 16;

/*
 * Prefetch a vector onto device.
 */
template<class T>
void prefetch(const T* x, const int n, const int incx) {
  ///@todo Currently disabled, performance worse
  //CUDA_CHECK(cudaMemPrefetchAsync(x, n*incx*sizeof(T), device, stream));
}

/*
 * Prefetch a matrix onto device.
 */
template<class T>
void prefetch(const T* A, const int m, const int n, const int ldA) {
  ///@todo Currently disabled, performance worse
  //CUDA_CHECK(cudaMemPrefetchAsync(A, n*ldA*sizeof(T), device, stream));
}

/*
 * Vector unary transform.
 */
template<class T, class Functor>
__global__ void kernel_transform(const int n, const T* x, const int incx,
    T* y, const int incy, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    y[i*incy] = f(x[i*incx]);
  }
}
template<class T, class Functor>
void transform(const int n, const T* x, const int incx, T* y, const int incy,
    Functor f) {
  auto grid = make_grid(n);
  auto block = make_block(n);
  kernel_transform<<<grid,block,0,stream>>>(n, x, incx, y, incy, f);
}

/*
 * Matrix unary transform.
 */
template<class T, class Functor>
__global__ void kernel_transform(const int m, const int n, const T* A,
    const int ldA, T* B, const int ldB, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    B[i + j*ldB] = f(A[i + j*ldA]);
  }
}
template<class T, class Functor>
void transform(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB, Functor f) {
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, A, ldA, B, ldB, f);
}

/*
 * Vector binary transform.
 */
template<class T, class Functor>
__global__ void kernel_transform(const int n, const T* x, const int incx,
    const T* y, const int incy, T* z, const int incz, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) {
    z[i*incz] = f(x[i*incx], y[i*incy]);
  }
}
template<class T, class Functor>
void transform(const int n, const T* x, const int incx, const T* y,
    const int incy, T* z, const int incz, Functor f) {
  auto grid = make_grid(n);
  auto block = make_block(n);
  kernel_transform<<<grid,block,0,stream>>>(n, x, incx, y, incy, z, incz, f);
}

/*
 * Matrix binary transform.
 */
template<class T, class Functor>
__global__ void kernel_transform(const int m, const int n, const T* A,
    const int ldA, const T* B, const int ldB, T* C, const int ldC,
    Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    C[i + j*ldC] = f(A[i + j*ldA], B[i + j*ldB]);
  }
}
template<class T, class Functor>
void transform(const int m, const int n, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC, Functor f) {
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, A, ldA, B, ldB, C, ldC, f);
}

/*
 * Matrix transpose kernel.
 */
template<class T>
__global__ void kernel_transpose(const int m, const int n, const T x,
    const T* A, const int ldA, T* B, const int ldB) {
  __shared__ T tile[TRANSPOSE_SIZE][TRANSPOSE_SIZE + 1];
  // ^ +1 reduce shared memory bank conflicts

  auto i = blockIdx.y*blockDim.y + threadIdx.x;
  auto j = blockIdx.x*blockDim.x + threadIdx.y;
  if (i < n && j < m) {
    tile[threadIdx.x][threadIdx.y] = A[i + j*ldA];
  }
  __syncthreads();
  i = blockIdx.x*blockDim.x + threadIdx.x;
  j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    B[i + j*ldB] = x*tile[threadIdx.y][threadIdx.x];
  }
}

template<class T>
void neg(const int n, const T* x, const int incx, T* y, const int incy) {
  prefetch(x, n, incx);
  prefetch(y, n, incy);
  transform(n, x, incx, y, incy, negate_functor<T>());
}

template<class T>
void neg(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, negate_functor<T>());
}

template<class T>
void add(const int n, const T* x, const int incx, const T* y, const int incy,
    T* z, const int incz) {
  prefetch(x, n, incx);
  prefetch(y, n, incy);
  prefetch(z, n, incz);
  transform(n, x, incx, y, incy, z, incz, plus_functor<T>());
}

template<class T>
void add(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, plus_functor<T>());
}

template<class T>
void sub(const int n, const T* x, const int incx, const T* y, const int incy,
    T* z, const int incz) {
  prefetch(x, n, incx);
  prefetch(y, n, incy);
  prefetch(z, n, incz);
  transform(n, x, incx, y, incy, z, incz, minus_functor<T>());
}

template<class T>
void sub(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, minus_functor<T>());
}

template<class T>
void hadamard(const int n, const T* x, const int incx, const T* y,
    const int incy, T* z, const int incz) {
  prefetch(x, n, incx);
  prefetch(y, n, incy);
  prefetch(z, n, incz);
  transform(n, x, incx, y, incy, z, incz, multiplies_functor<T>());
}

template<class T>
void hadamard(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, multiplies_functor<T>());
}

template<class T>
void div(const int n, const T* x, const int incx, const T y, T* z,
    const int incz) {
  prefetch(x, n, incx);
  prefetch(z, n, incz);
  transform(n, x, incx, z, incz, scalar_divides_functor<T>(y));
}

template<class T>
void div(const int m, const int n, const T* A, const int ldA, const T b, T* C,
    const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, C, ldC, scalar_divides_functor<T>(b));
}

template<class T>
void mul(const int n, const T x, const T* y, const int incy, T* z,
    const int incz) {
  prefetch(y, n, incy);
  prefetch(z, n, incz);
  transform(n, y, incy, z, incz, scalar_multiplies_functor<T>(x));
}

template<class T>
void mul(const int m, const int n, const T a, const T* B, const int ldB, T* C,
    const int ldC) {
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, B, ldB, C, ldC, scalar_multiplies_functor<T>(a));
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
void cholmul(const int n, const T* S, const int ldS, const T* x,
    const int incx, T* y, const int incy) {
  prefetch(S, n, n, ldS);
  prefetch(x, n, incx);
  prefetch(y, n, incy);

  memcpy(y, incy*sizeof(T), x, incx*sizeof(T), sizeof(T), n);
  auto ldL = n;
  auto L = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
  memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), n*sizeof(T), n);

  /* Cholesky factorization */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L,
      ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L, ldL,
      cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
      bufferOnHostBytes, info));

  /* multiplication */
  CUBLAS_CHECK(cublas<T>::trmv(cublasHandle, CUBLAS_FILL_MODE_LOWER,
      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, L, ldL, y, incy));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

template<class T>
void cholmul(const int m, const int n, const T* S, const int ldS, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(S, m, m, ldS);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);

  auto ldL = m;
  auto L = (T*)device_malloc(sizeof(T)*std::max(1, m*m));
  memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), m*sizeof(T), m);

  /* Cholesky factorization */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, m, cusolver<T>::CUDA_R, L,
      ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, m, cusolver<T>::CUDA_R, L, ldL,
      cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
      bufferOnHostBytes, info));

  /* multiplication */
  CUBLAS_CHECK(cublas<T>::trmm(cublasHandle, CUBLAS_SIDE_LEFT,
      CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n,
      scalar<T>::one, L, ldL, B, ldB, C, ldC));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

template<class T>
T sum(const int n, const T* x, const int incx) {
  prefetch(x, n, incx);

  T* y = (T*)malloc(sizeof(T));
  auto x1 = make_cub_vector(x, n, incx);
  void* tmp = nullptr;
  size_t bytes = 0;

  CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, x1.begin(), y, n, stream));
  tmp = device_malloc(bytes);
  CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, x1.begin(), y, n, stream));
  wait();
  T z = *y;

  device_free(tmp);
  free(y);
  return z;
}

template<class T>
T sum(const int m, const int n, const T* A, const int ldA) {
  prefetch(A, m, n, ldA);

  T* y = (T*)malloc(sizeof(T));
  auto A1 = make_cub_matrix(A, m, n, ldA);
  void* tmp = nullptr;
  size_t bytes = 0;

  CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, A1.begin(), y, m*n, stream));
  tmp = device_malloc(bytes);
  CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, A1.begin(), y, m*n, stream));
  wait();
  T z = *y;

  device_free(tmp);
  free(y);
  return z;
}

template<class T>
T dot(const int n, const T* x, const int incx, const T* y, const int incy) {
  prefetch(x, n, incx);
  prefetch(y, n, incy);

  T* z = (T*)malloc(sizeof(T));
  CUBLAS_CHECK(cublas<T>::dot(cublasHandle, n, x, incx, y, incy, z));
  wait();
  T result = *z;
  free(z);
  return result;
}

template<class T>
T frobenius(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);

  ///@todo Remove temporary
  auto C = (T*)device_malloc(m*n*sizeof(T));
  auto ldC = m;
  hadamard(m, n, A, ldA, B, ldB, C, ldC);
  auto z = sum(m, n, C, ldC);
  device_free(C);
  return z;
}

template<class T>
void inner(const int m, const int n, const T* A, const int ldA, const T* x,
    const int incx, T* y, const int incy) {
  prefetch(A, n, m, ldA);
  prefetch(x, n, incx);
  prefetch(y, m, incy);
  CUBLAS_CHECK(cublas<T>::gemv(cublasHandle, CUBLAS_OP_T, n, m,
      scalar<T>::one, A, ldA, x, incx, scalar<T>::zero, y, incy));
}

template<class T>
void inner(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  prefetch(A, k, m, ldA);
  prefetch(B, k, n, ldB);
  prefetch(C, m, n, ldC);
  CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n,
      k, scalar<T>::one, A, ldA, B, ldB, scalar<T>::zero, C, ldC));
}

template<class T>
void outer(const int m, const int n, const T* x, const int incx, const T* y,
    const int incy, T* A, const int ldA) {
  prefetch(x, m, incx);
  prefetch(y, n, incy);
  prefetch(A, m, n, ldA);

  /* here, the two vectors are interpreted as single-row matrices, so that the
   * stride between elements becomes the stride between columns; to create the
   * outer product, the first matrix is transposed to a single-column matrix,
   * while the second is not */
  CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n,
      1, scalar<T>::one, x, incx, y, incy, scalar<T>::zero, A, ldA));
}

template<class T>
void outer(const int m, const int n, const int k, const T* A, const int ldA,
    const T* B, const int ldB, T* C, const int ldC) {
  prefetch(A, m, k, ldA);
  prefetch(B, n, k, ldB);
  prefetch(C, m, n, ldC);
  CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n,
      k, scalar<T>::one, A, ldA, B, ldB, scalar<T>::zero, C, ldC));
}

template<class T>
void cholouter(const int m, const int n, const T* A, const int ldA,
    const T* S, const int ldS, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(S, n, n, ldS);
  prefetch(C, m, n, ldC);

  auto ldL = n;
  auto L = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
  memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), n*sizeof(T), n);

  /* Cholesky factorization */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_UPPER, n, cusolver<T>::CUDA_R, L,
      ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_UPPER, n, cusolver<T>::CUDA_R, L, ldL,
      cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
      bufferOnHostBytes, info));

  /* multiplication */
  CUBLAS_CHECK(cublas<T>::trmm(cublasHandle, CUBLAS_SIDE_RIGHT,
      CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n,
      scalar<T>::one, L, ldL, A, ldA, C, ldC));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

template<class T>
void solve(const int n, const T* A, const int ldA, T* x, const int incx,
    const T* y, const int incy) {
  prefetch(A, n, n, ldA);
  prefetch(x, n, incx);
  prefetch(y, n, incy);

  auto LU = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
  auto ldLU = n;
  memcpy(LU, ldLU*sizeof(T), A, ldA*sizeof(T), n*sizeof(T), n);

  auto x1 = x;
  if (incx > 1) {
    x1 = (T*)device_malloc(sizeof(T)*n);
  }
  memcpy(x1, sizeof(T), y, incy*sizeof(T), sizeof(T), n);

  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, n, n, cusolver<T>::CUDA_R, LU, ldLU,
      cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = malloc(bufferOnHostBytes);
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*std::max(1, n));

  /* solve via LU factorization with partial pivoting */
  CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n,
      n, cusolver<T>::CUDA_R, LU, ldLU, ipiv, cusolver<T>::CUDA_R,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));
  CUSOLVER_CHECK_INFO(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_OP_N, n, 1, cusolver<T>::CUDA_R, LU, ldLU, ipiv,
      cusolver<T>::CUDA_R, x1, n, info));
  if (incx > 1) {
    memcpy(x, incx*sizeof(T), x1, sizeof(T), sizeof(T), n);
  }

  device_free(ipiv);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  if (incx > 1) {
    device_free(x1);
  }
  device_free(LU);
}

template<class T>
void solve(const int m, const int n, const T* A, const int ldA, T* X,
    const int ldX, const T* Y, const int ldY) {
  prefetch(A, m, m, ldA);
  prefetch(X, m, n, ldX);
  prefetch(Y, m, n, ldY);

  memcpy(X, ldX*sizeof(T), Y, ldY*sizeof(T), m*sizeof(T), n);
  auto LU = (T*)device_malloc(sizeof(T)*std::max(1, m*m));
  auto ldLU = m;
  memcpy(LU, ldLU*sizeof(T), A, ldA*sizeof(T), n*sizeof(T), n);

  /* solve via LU factorization with partial pivoting */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, m, m, cusolver<T>::CUDA_R, LU, ldLU,
      cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*std::min(m, n));

  CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n,
      n, cusolver<T>::CUDA_R, LU, ldLU, ipiv, cusolver<T>::CUDA_R,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));
  CUSOLVER_CHECK_INFO(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_OP_N, m, n, cusolver<T>::CUDA_R, LU, ldLU, ipiv,
      cusolver<T>::CUDA_R, X, ldX, info));

  device_free(ipiv);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(LU);
}

template<class T>
void cholsolve(const int n, const T* S, const int ldS, T* x, const int incx,
    const T* y, const int incy) {
  prefetch(S, n, n, ldS);
  prefetch(x, n, incx);
  prefetch(y, n, incy);

  T* x1 = x;
  if (incx > 1) {
    x1 = (T*)device_malloc(n*sizeof(T));
  }
  memcpy(x1, sizeof(T), y, incy*sizeof(T), sizeof(T), n);
  auto L = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
  auto ldL = n;
  memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), n*sizeof(T), n);

  /* solve via Cholesky factorization */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L,
      ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L, ldL,
      cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
      bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, 1, cusolver<T>::CUDA_R, L, ldL,
      cusolver<T>::CUDA_R, x1, n, info));
  if (incx > 1) {
    memcpy(x, incx*sizeof(T), x1, sizeof(T), sizeof(T), n);
  }

  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
  if (incx > 1) {
    device_free(x1);
  }
}

template<class T>
void cholsolve(const int m, const int n, const T* S, const int ldS, T* X,
    const int ldX, const T* Y, const int ldY) {
  prefetch(S, m, m, ldS);
  prefetch(X, m, n, ldX);
  prefetch(Y, m, n, ldY);

  memcpy(X, ldX*sizeof(T), Y, ldY*sizeof(T), m*sizeof(T), n);
  auto L = (T*)device_malloc(sizeof(T)*std::max(1, m*m));
  auto ldL = m;
  memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), m*sizeof(T), m);

  /* solve via Cholesky factorization */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, m, cusolver<T>::CUDA_R, L,
      ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, m, cusolver<T>::CUDA_R, L, ldL,
      cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
      bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, m, n, cusolver<T>::CUDA_R, L, ldL,
      cusolver<T>::CUDA_R, X, ldX, info));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

template<class T>
void inv(const int n, const T* A, const int ldA, T* B, const int ldB) {
  prefetch(A, n, n, ldA);
  prefetch(B, n, n, ldB);

  /* write identity matrix into B */
  CUDA_CHECK(cudaMemset2DAsync(B, ldB*sizeof(T), 0, n*sizeof(T), n, stream));
  CUBLAS_CHECK(cublas<T>::copy(cublasHandle, n, scalar<T>::one, 0, B,
      ldB + 1));

  auto LU = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
  auto ldLU = n;
  memcpy(LU, ldLU*sizeof(T), A, ldA*sizeof(T), n*sizeof(T), n);

  /* solve via LU factorization with partial pivoting */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, n, n, cusolver<T>::CUDA_R, LU, ldLU,
      cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = malloc(bufferOnHostBytes);
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*n);

  CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n,
      n, cusolver<T>::CUDA_R, LU, ldLU, ipiv, cusolver<T>::CUDA_R,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));
  CUSOLVER_CHECK_INFO(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_OP_N, n, n, cusolver<T>::CUDA_R, LU, ldLU, ipiv,
      cusolver<T>::CUDA_R, B, ldB, info));

  device_free(ipiv);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(LU);
}

template<class T>
void cholinv(const int n, const T* S, const int ldS, T* B, const int ldB) {
  prefetch(S, n, n, ldS);
  prefetch(B, n, n, ldB);

  /* write identity matrix into B */
  CUDA_CHECK(cudaMemset2DAsync(B, ldB*sizeof(T), 0, n*sizeof(T), n, stream));
  CUBLAS_CHECK(cublas<T>::copy(cublasHandle, n, scalar<T>::one, 0, B,
      ldB + 1));

  auto L = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
  auto ldL = n;
  memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), n*sizeof(T), n);

  /* solve via Cholesky factorization */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L,
      ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L, ldL,
      cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
      bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, n, cusolver<T>::CUDA_R, L, ldL,
      cusolver<T>::CUDA_R, B, ldB, info));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

template<class T>
T ldet(const int n, const T* A, const int ldA) {
  prefetch(A, n, n, ldA);

  auto LU = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
  auto ldLU = n;
  memcpy(LU, ldLU*sizeof(T), A, ldA*sizeof(T), n*sizeof(T), n);

  /* LU factorization with partial pivoting */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, n, n, cusolver<T>::CUDA_R, LU, ldLU,
      cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*n);

  CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n,
      n, cusolver<T>::CUDA_R, LU, ldLU, nullptr/*ipiv*/, cusolver<T>::CUDA_R,
      bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
      info));

  /* the LU factorization is with partial pivoting, which means $|A| = (-1)^p
   * |L||U|$, where $p$ is the number of row exchanges in `ipiv`; however,
   * we're taking the logarithm of its absolute value, so can ignore the first
   * term, and the second term is just 1 as $L$ has a unit diagonal; just need
   * $|U|$ here; the logarithm of its absolute value is just the sum of the
   * logarithms of the absolute values of elements on the main diagonal */
  ///@todo Remove temporary
  auto d = (T*)device_malloc(n*sizeof(T));
  transform(n, LU, ldLU + 1, d, 1, log_abs_functor<T>());
  T ldet = sum(n, d, 1);

  device_free(d);
  device_free(ipiv);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(LU);

  return ldet;
}

template<class T>
T lcholdet(const int n, const T* S, const int ldS) {
  prefetch(S, n, n, ldS);

  auto L = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
  auto ldL = n;
  memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), n*sizeof(T), n);

  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L,
      ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);

  /* solve via Cholesky factorization */
  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L, ldL,
      cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
      bufferOnHostBytes, info));

  /* log-determinant is twice the sum of logarithms of elements on the main
   * diagonal, all of which should be positive */
  ///@todo Remove temporary
  auto d = (T*)device_malloc(n*sizeof(T));
  transform(n, L, ldL + 1, d, 1, log_functor<T>());
  T ldet = 2.0*sum(n, d, 1);

  device_free(d);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);

  return ldet;
}

template<class T>
void transpose(const int m, const int n, const T x, const T* A, const int ldA,
    T* B, const int ldB) {
  prefetch(A, n, m, ldA);
  prefetch(B, m, n, ldB);

  dim3 block;
  block.x = TRANSPOSE_SIZE;
  block.y = TRANSPOSE_SIZE;
  block.z = 1;

  dim3 grid;
  grid.x = (m + TRANSPOSE_SIZE - 1)/TRANSPOSE_SIZE;
  grid.y = (n + TRANSPOSE_SIZE - 1)/TRANSPOSE_SIZE;
  grid.z = 1;

  auto shared = TRANSPOSE_SIZE*TRANSPOSE_SIZE*sizeof(T);
  kernel_transpose<<<grid,block,shared,stream>>>(m, n, x, A, ldA, B, ldB);
}

template<class T>
T trace(const int m, const int n, const T* A, const int ldA) {
  return sum(std::min(m, n), A, ldA + 1);
}

}
