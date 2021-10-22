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
void copysign(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, copysign_functor<T>());
}

template<class T>
void digamma(const int m, const int n, const T* A, const int ldA,
    const int* B, const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, digammap_functor<T>());
}

template<class T>
void dot(const int n, const T* x, const int incx, const T* y, const int incy,
    T* z) {
  prefetch(x, n, incx);
  prefetch(y, n, incy);

  CUBLAS_CHECK(cublas<T>::dot(cublasHandle, n, x, incx, y, incy, z));
}

template<class T>
void frobenius(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* c) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);

  ///@todo Remove temporary
  auto C = (T*)device_malloc(m*n*sizeof(T));
  auto ldC = m;
  hadamard(m, n, A, ldA, B, ldB, C, ldC);
  sum(m, n, C, ldC, c);
  device_free(C);
}

template<class T>
void gamma_p(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, gamma_p_functor<T>());
}

template<class T>
void gamma_q(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, gamma_q_functor<T>());
}

template<class T>
void hadamard(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, multiply_functor<T>());
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
void lbeta(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, lbeta_functor<T>());
}

template<class T>
void lchoose(const int m, const int n, const int* A, const int ldA,
    const int* B, const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, lchoose_functor<T>());
}

template<class T>
void lgamma(const int m, const int n, const T* A, const int ldA, const int* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, lgammap_functor<T>());
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
void pow(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, C, ldC, pow_functor<T>());
}

template<class T>
void single(const int* i, const int* j, const int m, const int n, T* A,
    const int ldA) {
  for_each(m, n, A, ldA, single_functor<T>(i, j));
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

}
