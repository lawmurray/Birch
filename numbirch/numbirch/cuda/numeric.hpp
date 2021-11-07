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

template<class T, class>
T operator+(const T& x) {
  return x;
}

template<class T, class>
T operator-(const T& x) {
  prefetch(x);
  return transform(x, negate_functor());
}

template<class T, class U, class>
promote_t<T,U> operator+(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, add_functor());
}

template<class T, class U, class>
promote_t<T,U> operator-(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, subtract_functor());
}

template<class T, class U, class>
promote_t<T,U> operator*(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, multiply_functor());
}

template<class T, class U, class>
promote_t<T,U> operator/(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, divide_functor());
}

template<class T, class U, class>
T operator!(const T& x) {
  prefetch(x);
  return transform(x, not_functor());
}

template<class T, class U, class>
Array<bool,dimension_v<T>> operator&&(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, and_functor());
}

template<class T, class U, class>
Array<bool,dimension_v<T>> operator||(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, or_functor());
}

template<class T, class U, class>
Array<bool,dimension_v<T>> operator==(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, equal_functor());
}

template<class T, class U, class>
Array<bool,dimension_v<T>> operator!=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, not_equal_functor());
}

template<class T, class U, class>
Array<bool,dimension_v<T>> operator<(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, less_functor());
}

template<class T, class U, class>
Array<bool,dimension_v<T>> operator<=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, less_or_equal_functor());
}

template<class T, class U, class>
Array<bool,dimension_v<T>> operator>(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, greater_functor());
}

template<class T, class U, class>
Array<bool,dimension_v<T>> operator>=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, greater_or_equal_functor());
}

// template<class T>
// void mul(const int m, const int n, const T* A, const int ldA, const T* x,
//     const int incx, T* y, const int incy) {
//   prefetch(A, m, n, ldA);
//   prefetch(x, n, incx);
//   prefetch(y, m, incy);
//   CUBLAS_CHECK(cublas<T>::gemv(cublasHandle, CUBLAS_OP_N, m, n,
//       scalar<T>::one, A, ldA, x, incx, scalar<T>::zero, y, incy));
// }

// template<class T>
// void mul(const int m, const int n, const int k, const T* A, const int ldA,
//     const T* B, const int ldB, T* C, const int ldC) {
//   prefetch(A, m, k, ldA);
//   prefetch(B, k, n, ldB);
//   prefetch(C, m, n, ldC);
//   CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
//       k, scalar<T>::one, A, ldA, B, ldB, scalar<T>::zero, C, ldC));
// }

// template<class T>
// void abs(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, abs_functor<T>());
// }

// template<class T>
// void acos(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, acos_functor<T>());
// }

// template<class T>
// void asin(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, asin_functor<T>());
// }

// template<class T>
// void atan(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, atan_functor<T>());
// }

// template<class T>
// void ceil(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, ceil_functor<T>());
// }

// template<class T>
// void cholinv(const int n, const T* S, const int ldS, T* B, const int ldB) {
//   prefetch(S, n, n, ldS);
//   prefetch(B, n, n, ldB);

//   /* write identity matrix into B */
//   CUDA_CHECK(cudaMemset2DAsync(B, ldB*sizeof(T), 0, n*sizeof(T), n, stream));
//   CUBLAS_CHECK(cublas<T>::copy(cublasHandle, n, scalar<T>::one, 0, B,
//       ldB + 1));

//   auto L = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
//   auto ldL = n;
//   memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), n*sizeof(T), n);

//   /* solve via Cholesky factorization */
//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L,
//       ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void* bufferOnHost = malloc(bufferOnHostBytes);

//   CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L, ldL,
//       cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
//       bufferOnHostBytes, info));
//   CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_FILL_MODE_LOWER, n, n, cusolver<T>::CUDA_R, L, ldL,
//       cusolver<T>::CUDA_R, B, ldB, info));

//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   device_free(L);
// }

// template<class T>
// void cos(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, cos_functor<T>());
// }

// template<class T>
// void cosh(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, cosh_functor<T>());
// }

// template<class T>
// void count(const int m, const int n, const T* A, const int ldA, int* b) {
//   prefetch(A, m, n, ldA);

//   ///@todo Remove temporary
//   auto B = (int*)device_malloc(m*n*sizeof(int));
//   int ldB = m;
//   transform(m, n, A, ldA, B, ldB, count_functor<T>());
//   sum(m, n, B, ldB, b);
//   device_free(B);
// }

// template<class T>
// void diagonal(const T* a, const int n, T* B, const int ldB) {
//   for_each(n, n, B, ldB, diagonal_functor<T>(a));
// }

// template<class T>
// void digamma(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, digamma_functor<T>());
// }

// template<class T>
// void exp(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, exp_functor<T>());
// }

// template<class T>
// void expm1(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, expm1_functor<T>());
// }

// template<class T>
// void floor(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, floor_functor<T>());
// }

// template<class T>
// void inv(const int n, const T* A, const int ldA, T* B, const int ldB) {
//   prefetch(A, n, n, ldA);
//   prefetch(B, n, n, ldB);

//   /* write identity matrix into B */
//   CUDA_CHECK(cudaMemset2DAsync(B, ldB*sizeof(T), 0, n*sizeof(T), n, stream));
//   CUBLAS_CHECK(cublas<T>::copy(cublasHandle, n, scalar<T>::one, 0, B,
//       ldB + 1));

//   auto LU = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
//   auto ldLU = n;
//   memcpy(LU, ldLU*sizeof(T), A, ldA*sizeof(T), n*sizeof(T), n);

//   /* solve via LU factorization with partial pivoting */
//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, n, n, cusolver<T>::CUDA_R, LU, ldLU,
//       cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void *bufferOnHost = malloc(bufferOnHostBytes);
//   auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*n);

//   CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n,
//       n, cusolver<T>::CUDA_R, LU, ldLU, ipiv, cusolver<T>::CUDA_R,
//       bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
//       info));
//   CUSOLVER_CHECK_INFO(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_OP_N, n, n, cusolver<T>::CUDA_R, LU, ldLU, ipiv,
//       cusolver<T>::CUDA_R, B, ldB, info));

//   device_free(ipiv);
//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   device_free(LU);
// }

// template<class T>
// void lcholdet(const int n, const T* S, const int ldS, T* b) {
//   prefetch(S, n, n, ldS);

//   auto L = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
//   auto ldL = n;
//   memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), n*sizeof(T), n);

//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L,
//       ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void* bufferOnHost = malloc(bufferOnHostBytes);

//   /* solve via Cholesky factorization */
//   CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L, ldL,
//       cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
//       bufferOnHostBytes, info));

//   /* log-determinant is twice the sum of logarithms of elements on the main
//    * diagonal, all of which should be positive */
//   ///@todo Remove temporary
//   auto d = (T*)device_malloc(n*sizeof(T));
//   transform(1, n, L, ldL + 1, d, 1, log_square_functor<T>());
//   sum(1, n, d, 1, b);

//   device_free(d);
//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   device_free(L);
// }

// template<class T>
// void ldet(const int n, const T* A, const int ldA, T* b) {
//   prefetch(A, n, n, ldA);

//   auto LU = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
//   auto ldLU = n;
//   memcpy(LU, ldLU*sizeof(T), A, ldA*sizeof(T), n*sizeof(T), n);

//   /* LU factorization with partial pivoting */
//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, n, n, cusolver<T>::CUDA_R, LU, ldLU,
//       cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void* bufferOnHost = malloc(bufferOnHostBytes);
//   auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*n);

//   CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n,
//       n, cusolver<T>::CUDA_R, LU, ldLU, nullptr/*ipiv*/, cusolver<T>::CUDA_R,
//       bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
//       info));

//   /* the LU factorization is with partial pivoting, which means $|A| = (-1)^p
//    * |L||U|$, where $p$ is the number of row exchanges in `ipiv`; however,
//    * we're taking the logarithm of its absolute value, so can ignore the first
//    * term, and the second term is just 1 as $L$ has a unit diagonal; just need
//    * $|U|$ here; the logarithm of its absolute value is just the sum of the
//    * logarithms of the absolute values of elements on the main diagonal */
//   ///@todo Remove temporary
//   auto d = (T*)device_malloc(n*sizeof(T));
//   transform(1, n, LU, ldLU + 1, d, 1, log_abs_functor<T>());
//   sum(1, n, d, 1, b);

//   device_free(d);
//   device_free(ipiv);
//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   device_free(LU);
// }

// template<class T>
// void lfact(const int m, const int n, const int* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, lfact_functor<T>());
// }

// template<class T>
// void lfact_grad(const int m, const int n, const T* G, const int ldG,
//     const int* A, const int ldA, T* B, const int ldB) {
//   prefetch(G, m, n, ldG);
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, G, ldG, A, ldA, B, ldB, lfact_grad_functor<T>());
// }

// template<class T>
// void lgamma(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, lgamma_functor<T>());
// }

// template<class T>
// void log(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, log_functor<T>());
// }

// template<class T>
// void log1p(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, log1p_functor<T>());
// }

// template<class T>
// void rcp(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, rcp_functor<T>());
// }

// template<class T>
// void rectify(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, rectify_functor<T>());
// }

// template<class T>
// void rectify_grad(const int m, const int n, const T* G, const int ldG,
//     const T* A, const int ldA, T* B, const int ldB) {
//   prefetch(G, m, n, ldG);
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, G, ldG, A, ldA, B, ldB, rectify_grad_functor<T>());
// }

// template<class T>
// void round(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, round_functor<T>());
// }

// template<class T>
// void sin(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, sin_functor<T>());
// }

// template<class T>
// void single(const int* i, const int n, T* x, const int incx) {
//   for_each(1, n, x, incx, single_functor<T>(scalar<int>::one, i));
// }

// template<class T>
// void sinh(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, sinh_functor<T>());
// }

// template<class T>
// void sqrt(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, sqrt_functor<T>());
// }

// template<class T>
// void sum(const int m, const int n, const T* A, const int ldA, T* b) {
//   prefetch(A, m, n, ldA);

//   auto A1 = make_cub_matrix(A, m, n, ldA);
//   void* tmp = nullptr;
//   size_t bytes = 0;

//   CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, A1.begin(), b, m*n, stream));
//   tmp = device_malloc(bytes);
//   CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, A1.begin(), b, m*n, stream));
//   device_free(tmp);
// }

// template<class T>
// void tan(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, tan_functor<T>());
// }

// template<class T>
// void tanh(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   transform(m, n, A, ldA, B, ldB, tanh_functor<T>());
// }

// template<class T>
// void trace(const int m, const int n, const T* A, const int ldA, T* b) {
//   return sum(1, std::min(m, n), A, ldA + 1, b);
// }

// template<class T>
// void transpose(const int m, const int n, const T* A, const int ldA, T* B,
//     const int ldB) {
//   prefetch(A, n, m, ldA);
//   prefetch(B, m, n, ldB);

//   dim3 block;
//   block.x = TRANSPOSE_SIZE;
//   block.y = TRANSPOSE_SIZE;
//   block.z = 1;

//   dim3 grid;
//   grid.x = (m + TRANSPOSE_SIZE - 1)/TRANSPOSE_SIZE;
//   grid.y = (n + TRANSPOSE_SIZE - 1)/TRANSPOSE_SIZE;
//   grid.z = 1;

//   auto shared = TRANSPOSE_SIZE*TRANSPOSE_SIZE*sizeof(T);
//   kernel_transpose<<<grid,block,shared,stream>>>(m, n, A, ldA, B, ldB);
// }

// template<class T>
// void cholmul(const int n, const T* S, const int ldS, const T* x,
//     const int incx, T* y, const int incy) {
//   prefetch(S, n, n, ldS);
//   prefetch(x, n, incx);
//   prefetch(y, n, incy);

//   memcpy(y, incy*sizeof(T), x, incx*sizeof(T), sizeof(T), n);
//   auto ldL = n;
//   auto L = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
//   memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), n*sizeof(T), n);

//   /* Cholesky factorization */
//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L,
//       ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void *bufferOnHost = malloc(bufferOnHostBytes);

//   CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L, ldL,
//       cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
//       bufferOnHostBytes, info));

//   /* multiplication */
//   CUBLAS_CHECK(cublas<T>::trmv(cublasHandle, CUBLAS_FILL_MODE_LOWER,
//       CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, L, ldL, y, incy));

//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   device_free(L);
// }

// template<class T>
// void cholmul(const int m, const int n, const T* S, const int ldS, const T* B,
//     const int ldB, T* C, const int ldC) {
//   prefetch(S, m, m, ldS);
//   prefetch(B, m, n, ldB);
//   prefetch(C, m, n, ldC);

//   auto ldL = m;
//   auto L = (T*)device_malloc(sizeof(T)*std::max(1, m*m));
//   memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), m*sizeof(T), m);

//   /* Cholesky factorization */
//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, CUBLAS_FILL_MODE_LOWER, m, cusolver<T>::CUDA_R, L,
//       ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void* bufferOnHost = malloc(bufferOnHostBytes);

//   CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_FILL_MODE_LOWER, m, cusolver<T>::CUDA_R, L, ldL,
//       cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
//       bufferOnHostBytes, info));

//   /* multiplication */
//   CUBLAS_CHECK(cublas<T>::trmm(cublasHandle, CUBLAS_SIDE_LEFT,
//       CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n,
//       scalar<T>::one, L, ldL, B, ldB, C, ldC));

//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   device_free(L);
// }

// template<class T>
// void cholouter(const int m, const int n, const T* A, const int ldA,
//     const T* S, const int ldS, T* C, const int ldC) {
//   prefetch(A, m, n, ldA);
//   prefetch(S, n, n, ldS);
//   prefetch(C, m, n, ldC);

//   auto ldL = n;
//   auto L = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
//   memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), n*sizeof(T), n);

//   /* Cholesky factorization */
//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, CUBLAS_FILL_MODE_UPPER, n, cusolver<T>::CUDA_R, L,
//       ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void* bufferOnHost = malloc(bufferOnHostBytes);

//   CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_FILL_MODE_UPPER, n, cusolver<T>::CUDA_R, L, ldL,
//       cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
//       bufferOnHostBytes, info));

//   /* multiplication */
//   CUBLAS_CHECK(cublas<T>::trmm(cublasHandle, CUBLAS_SIDE_RIGHT,
//       CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n,
//       scalar<T>::one, L, ldL, A, ldA, C, ldC));

//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   device_free(L);
// }

// template<class T>
// void cholsolve(const int n, const T* S, const int ldS, T* x, const int incx,
//     const T* y, const int incy) {
//   prefetch(S, n, n, ldS);
//   prefetch(x, n, incx);
//   prefetch(y, n, incy);

//   T* x1 = x;
//   if (incx > 1) {
//     x1 = (T*)device_malloc(n*sizeof(T));
//   }
//   memcpy(x1, sizeof(T), y, incy*sizeof(T), sizeof(T), n);
//   auto L = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
//   auto ldL = n;
//   memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), n*sizeof(T), n);

//   /* solve via Cholesky factorization */
//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L,
//       ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void *bufferOnHost = malloc(bufferOnHostBytes);

//   CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_FILL_MODE_LOWER, n, cusolver<T>::CUDA_R, L, ldL,
//       cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
//       bufferOnHostBytes, info));
//   CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_FILL_MODE_LOWER, n, 1, cusolver<T>::CUDA_R, L, ldL,
//       cusolver<T>::CUDA_R, x1, n, info));
//   if (incx > 1) {
//     memcpy(x, incx*sizeof(T), x1, sizeof(T), sizeof(T), n);
//   }

//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   device_free(L);
//   if (incx > 1) {
//     device_free(x1);
//   }
// }

// template<class T>
// void cholsolve(const int m, const int n, const T* S, const int ldS, T* X,
//     const int ldX, const T* Y, const int ldY) {
//   prefetch(S, m, m, ldS);
//   prefetch(X, m, n, ldX);
//   prefetch(Y, m, n, ldY);

//   memcpy(X, ldX*sizeof(T), Y, ldY*sizeof(T), m*sizeof(T), n);
//   auto L = (T*)device_malloc(sizeof(T)*std::max(1, m*m));
//   auto ldL = m;
//   memcpy(L, ldL*sizeof(T), S, ldS*sizeof(T), m*sizeof(T), m);

//   /* solve via Cholesky factorization */
//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, CUBLAS_FILL_MODE_LOWER, m, cusolver<T>::CUDA_R, L,
//       ldL, cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void* bufferOnHost = malloc(bufferOnHostBytes);

//   CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_FILL_MODE_LOWER, m, cusolver<T>::CUDA_R, L, ldL,
//       cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
//       bufferOnHostBytes, info));
//   CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_FILL_MODE_LOWER, m, n, cusolver<T>::CUDA_R, L, ldL,
//       cusolver<T>::CUDA_R, X, ldX, info));

//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   device_free(L);
// }

// template<class T>
// void copysign(const int m, const int n, const T* A, const int ldA, const T* B,
//     const int ldB, T* C, const int ldC) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(C, m, n, ldC);
//   transform(m, n, A, ldA, B, ldB, C, ldC, copysign_functor<T>());
// }

// template<class T>
// void digamma(const int m, const int n, const T* A, const int ldA,
//     const int* B, const int ldB, T* C, const int ldC) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(C, m, n, ldC);
//   transform(m, n, A, ldA, B, ldB, C, ldC, digammap_functor<T>());
// }

// template<class T>
// void dot(const int n, const T* x, const int incx, const T* y, const int incy,
//     T* z) {
//   prefetch(x, n, incx);
//   prefetch(y, n, incy);

//   CUBLAS_CHECK(cublas<T>::dot(cublasHandle, n, x, incx, y, incy, z));
// }

// template<class T>
// void frobenius(const int m, const int n, const T* A, const int ldA, const T* B,
//     const int ldB, T* c) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);

//   ///@todo Remove temporary
//   auto C = (T*)device_malloc(m*n*sizeof(T));
//   auto ldC = m;
//   hadamard(m, n, A, ldA, B, ldB, C, ldC);
//   sum(m, n, C, ldC, c);
//   device_free(C);
// }

// template<class T>
// void gamma_p(const int m, const int n, const T* A, const int ldA, const T* B,
//     const int ldB, T* C, const int ldC) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(C, m, n, ldC);
//   transform(m, n, A, ldA, B, ldB, C, ldC, gamma_p_functor<T>());
// }

// template<class T>
// void gamma_q(const int m, const int n, const T* A, const int ldA, const T* B,
//     const int ldB, T* C, const int ldC) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(C, m, n, ldC);
//   transform(m, n, A, ldA, B, ldB, C, ldC, gamma_q_functor<T>());
// }

template<class T, class U, class>
typename promote<T,U>::type hadamard(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, multiply_functor());
}

// template<class T>
// void inner(const int m, const int n, const T* A, const int ldA, const T* x,
//     const int incx, T* y, const int incy) {
//   prefetch(A, n, m, ldA);
//   prefetch(x, n, incx);
//   prefetch(y, m, incy);
//   CUBLAS_CHECK(cublas<T>::gemv(cublasHandle, CUBLAS_OP_T, n, m,
//       scalar<T>::one, A, ldA, x, incx, scalar<T>::zero, y, incy));
// }

// template<class T>
// void inner(const int m, const int n, const int k, const T* A, const int ldA,
//     const T* B, const int ldB, T* C, const int ldC) {
//   prefetch(A, k, m, ldA);
//   prefetch(B, k, n, ldB);
//   prefetch(C, m, n, ldC);
//   CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n,
//       k, scalar<T>::one, A, ldA, B, ldB, scalar<T>::zero, C, ldC));
// }

// template<class T>
// void lbeta(const int m, const int n, const T* A, const int ldA, const T* B,
//     const int ldB, T* C, const int ldC) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(C, m, n, ldC);
//   transform(m, n, A, ldA, B, ldB, C, ldC, lbeta_functor<T>());
// }

// template<class T>
// void lchoose(const int m, const int n, const int* A, const int ldA,
//     const int* B, const int ldB, T* C, const int ldC) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(C, m, n, ldC);
//   transform(m, n, A, ldA, B, ldB, C, ldC, lchoose_functor<T>());
// }

// template<class T>
// void lchoose_grad(const int m, const int n, const T* G, const int ldG,
//     const int* A, const int ldA, const int* B, const int ldB, T* GA,
//     const int ldGA, T* GB, const int ldGB) {
//   prefetch(G, m, n, ldG);
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(GA, m, n, ldGA);
//   prefetch(GB, m, n, ldGB);
//   transform(m, n, G, ldG, A, ldA, B, ldB, GA, ldGA, GB, ldGB,
//       lchoose_grad_functor<T>());
// }

// template<class T>
// void lgamma(const int m, const int n, const T* A, const int ldA, const int* B,
//     const int ldB, T* C, const int ldC) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(C, m, n, ldC);
//   transform(m, n, A, ldA, B, ldB, C, ldC, lgammap_functor<T>());
// }

// template<class T>
// void outer(const int m, const int n, const T* x, const int incx, const T* y,
//     const int incy, T* A, const int ldA) {
//   prefetch(x, m, incx);
//   prefetch(y, n, incy);
//   prefetch(A, m, n, ldA);

//   /* here, the two vectors are interpreted as single-row matrices, so that the
//    * stride between elements becomes the stride between columns; to create the
//    * outer product, the first matrix is transposed to a single-column matrix,
//    * while the second is not */
//   CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, m, n,
//       1, scalar<T>::one, x, incx, y, incy, scalar<T>::zero, A, ldA));
// }

// template<class T>
// void outer(const int m, const int n, const int k, const T* A, const int ldA,
//     const T* B, const int ldB, T* C, const int ldC) {
//   prefetch(A, m, k, ldA);
//   prefetch(B, n, k, ldB);
//   prefetch(C, m, n, ldC);
//   CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, m, n,
//       k, scalar<T>::one, A, ldA, B, ldB, scalar<T>::zero, C, ldC));
// }

// template<class T>
// void pow(const int m, const int n, const T* A, const int ldA, const T* B,
//     const int ldB, T* C, const int ldC) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(C, m, n, ldC);
//   transform(m, n, A, ldA, B, ldB, C, ldC, pow_functor<T>());
// }

// template<class T>
// void single(const int* i, const int* j, const int m, const int n, T* A,
//     const int ldA) {
//   for_each(m, n, A, ldA, single_functor<T>(i, j));
// }

// template<class T>
// void solve(const int n, const T* A, const int ldA, T* x, const int incx,
//     const T* y, const int incy) {
//   prefetch(A, n, n, ldA);
//   prefetch(x, n, incx);
//   prefetch(y, n, incy);

//   auto LU = (T*)device_malloc(sizeof(T)*std::max(1, n*n));
//   auto ldLU = n;
//   memcpy(LU, ldLU*sizeof(T), A, ldA*sizeof(T), n*sizeof(T), n);

//   auto x1 = x;
//   if (incx > 1) {
//     x1 = (T*)device_malloc(sizeof(T)*n);
//   }
//   memcpy(x1, sizeof(T), y, incy*sizeof(T), sizeof(T), n);

//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, n, n, cusolver<T>::CUDA_R, LU, ldLU,
//       cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void *bufferOnHost = malloc(bufferOnHostBytes);
//   auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*std::max(1, n));

//   /* solve via LU factorization with partial pivoting */
//   CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n,
//       n, cusolver<T>::CUDA_R, LU, ldLU, ipiv, cusolver<T>::CUDA_R,
//       bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
//       info));
//   CUSOLVER_CHECK_INFO(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_OP_N, n, 1, cusolver<T>::CUDA_R, LU, ldLU, ipiv,
//       cusolver<T>::CUDA_R, x1, n, info));
//   if (incx > 1) {
//     memcpy(x, incx*sizeof(T), x1, sizeof(T), sizeof(T), n);
//   }

//   device_free(ipiv);
//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   if (incx > 1) {
//     device_free(x1);
//   }
//   device_free(LU);
// }

// template<class T>
// void solve(const int m, const int n, const T* A, const int ldA, T* X,
//     const int ldX, const T* Y, const int ldY) {
//   prefetch(A, m, m, ldA);
//   prefetch(X, m, n, ldX);
//   prefetch(Y, m, n, ldY);

//   memcpy(X, ldX*sizeof(T), Y, ldY*sizeof(T), m*sizeof(T), n);
//   auto LU = (T*)device_malloc(sizeof(T)*std::max(1, m*m));
//   auto ldLU = m;
//   memcpy(LU, ldLU*sizeof(T), A, ldA*sizeof(T), n*sizeof(T), n);

//   /* solve via LU factorization with partial pivoting */
//   size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
//   CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
//       cusolverDnParams, m, m, cusolver<T>::CUDA_R, LU, ldLU,
//       cusolver<T>::CUDA_R, &bufferOnDeviceBytes, &bufferOnHostBytes));
//   void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
//   void* bufferOnHost = malloc(bufferOnHostBytes);
//   auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*std::min(m, n));

//   CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams, n,
//       n, cusolver<T>::CUDA_R, LU, ldLU, ipiv, cusolver<T>::CUDA_R,
//       bufferOnDevice, bufferOnDeviceBytes, bufferOnHost, bufferOnHostBytes,
//       info));
//   CUSOLVER_CHECK_INFO(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
//       CUBLAS_OP_N, m, n, cusolver<T>::CUDA_R, LU, ldLU, ipiv,
//       cusolver<T>::CUDA_R, X, ldX, info));

//   device_free(ipiv);
//   free(bufferOnHost);
//   device_free(bufferOnDevice);
//   device_free(LU);
// }

// template<class T, class U>
// void ibeta(const int m, const int n, const U* A, const int ldA, const U* B,
//     const int ldB, const T* X, const int ldX, T* C, const int ldC) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(X, m, n, ldX);
//   prefetch(C, m, n, ldC);
//   transform(m, n, A, ldA, B, ldB, X, ldX, C, ldC, ibeta_functor<T>());
// }

// template<class T>
// void combine(const int m, const int n, const T a, const T* A, const int ldA,
//     const T b, const T* B, const int ldB, const T c, const T* C,
//     const int ldC, const T d, const T* D, const int ldD, T* E,
//     const int ldE) {
//   prefetch(A, m, n, ldA);
//   prefetch(B, m, n, ldB);
//   prefetch(C, m, n, ldC);
//   prefetch(D, m, n, ldD);
//   prefetch(E, m, n, ldE);

//   transform(m, n, A, ldA, B, ldB, C, ldC, D, ldD, E, ldE,
//       combine_functor<T>(a, b, c, d));
// }

}
