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
void abs(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, abs_functor<T>());
}

template<class T>
void acos(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, acos_functor<T>());
}

template<class T>
void asin(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, asin_functor<T>());
}

template<class T>
void atan(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, atan_functor<T>());
}

template<class T>
void ceil(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, ceil_functor<T>());
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
void cos(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, cos_functor<T>());
}

template<class T>
void cosh(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, cosh_functor<T>());
}

template<class T>
void exp(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, exp_functor<T>());
}

template<class T>
void expm1(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, expm1_functor<T>());
}

template<class T>
void floor(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, floor_functor<T>());
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
void lcholdet(const int n, const T* S, const int ldS, T* b) {
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
  transform(1, n, L, ldL + 1, d, 1, log_square_functor<T>());
  sum(1, n, d, 1, b);

  device_free(d);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(L);
}

template<class T>
void ldet(const int n, const T* A, const int ldA, T* b) {
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
  transform(1, n, LU, ldLU + 1, d, 1, log_abs_functor<T>());
  sum(1, n, d, 1, b);

  device_free(d);
  device_free(ipiv);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  device_free(LU);
}

template<class T>
void lgamma(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, lgamma_functor<T>());
}

template<class T>
void log(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, log_functor<T>());
}

template<class T>
void log1p(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, log1p_functor<T>());
}

template<class T>
void rectify(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, rectify_functor<T>());
}

template<class T>
void round(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, round_functor<T>());
}

template<class T>
void sin(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, sin_functor<T>());
}

template<class T>
void sinh(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, sinh_functor<T>());
}

template<class T>
void sqrt(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, sqrt_functor<T>());
}

template<class T>
void sum(const int m, const int n, const T* A, const int ldA, T* b) {
  prefetch(A, m, n, ldA);

  auto A1 = make_cub_matrix(A, m, n, ldA);
  void* tmp = nullptr;
  size_t bytes = 0;

  CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, A1.begin(), b, m*n, stream));
  tmp = device_malloc(bytes);
  CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, A1.begin(), b, m*n, stream));
  device_free(tmp);
}

template<class T>
void tan(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, tan_functor<T>());
}

template<class T>
void tanh(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, tanh_functor<T>());
}

template<class T>
void trace(const int m, const int n, const T* A, const int ldA, T* b) {
  return sum(1, std::min(m, n), A, ldA + 1, b);
}

template<class T>
void transpose(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
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
  kernel_transpose<<<grid,block,shared,stream>>>(m, n, A, ldA, B, ldB);
}

}
