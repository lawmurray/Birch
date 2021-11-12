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

template<class R, class T, class>
convert_t<R,T> operator+(const T& x) {
  if constexpr (std::is_same_v<R,T>) {
    return x;
  } else {
    return transform(x, identity_functor<R>());
  }
}

template<class R, class T, class>
convert_t<R,T> operator-(const T& x) {
  prefetch(x);
  return transform(x, negate_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator+(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, add_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator-(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, subtract_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator*(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, multiply_functor<R>());
}

template<class T, class>
Array<T,1> operator*(const Array<T,2>& A, const Array<T,1>& x) {
  assert(columns(A) == length(x));
  prefetch(A);
  prefetch(x);
  Array<T,1> y(make_shape(rows(A)));
  CUBLAS_CHECK(cublas<T>::gemv(cublasHandle, CUBLAS_OP_N, rows(A), columns(A),
      scalar<T>::one, data(A), stride(A), data(x), stride(x), scalar<T>::zero,
      data(y), stride(y)));
  return y;
}

template<class T, class>
Array<T,2> operator*(const Array<T,2>& A, const Array<T,2>& B) {
  assert(columns(A) == rows(B));
  prefetch(A);
  prefetch(B);
  Array<T,2> C(make_shape(rows(A), columns(B)));
  CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
      rows(C), columns(C), columns(A), scalar<T>::one, data(A), stride(A),
      data(B), stride(B), scalar<T>::zero, data(C), stride(C)));
  return C;
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator/(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, divide_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> operator!(const T& x) {
  prefetch(x);
  return transform(x, not_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator&&(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, and_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator||(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, or_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator==(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, equal_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator!=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, not_equal_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator<(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, less_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator<=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, less_or_equal_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator>(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, greater_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> operator>=(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, greater_or_equal_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> abs(const T& x) {
  prefetch(x);
  return transform(x, abs_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> acos(const T& x) {
  prefetch(x);
  return transform(x, acos_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> asin(const T& x) {
  prefetch(x);
  return transform(x, asin_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> atan(const T& x) {
  prefetch(x);
  return transform(x, atan_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> ceil(const T& x) {
  prefetch(x);
  return transform(x, ceil_functor<R>());
}

template<class T, class>
Array<T,2> cholinv(const Array<T,2>& S) {
  assert(rows(S) == columns(S));
  Array<T,2> L(S);
  Array<T,2> B(shape(S));

  /* write identity matrix into B */
  CUDA_CHECK(cudaMemset2DAsync(data(B), stride(B)*sizeof(T), 0,
      rows(B)*sizeof(T), columns(B), stream));
  CUBLAS_CHECK(cublas<T>::copy(cublasHandle, rows(B), scalar<T>::one, 0,
      data(B), stride(B) + 1));

  /* invert via Cholesky factorization */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R,
      data(L), stride(L), cusolver<T>::CUDA_R, &bufferOnDeviceBytes,
      &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(B), columns(B), cusolver<T>::CUDA_R,
      data(L), stride(L), cusolver<T>::CUDA_R, data(B), stride(B), info));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  return B;
}

template<class R, class T, class>
convert_t<R,T> cos(const T& x) {
  prefetch(x);
  return transform(x, cos_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> cosh(const T& x) {
  prefetch(x);
  return transform(x, cosh_functor<R>());
}

template<class R, class T, class>
Array<R,0> count(const T& x) {
  ///@todo Avoid temporary
  return sum(transform(x, count_functor<R>()));
}

template<class R, class T, class>
Array<R,2> diagonal(const T& x, const int n) {
  return for_each(n, n, diagonal_functor<R,decltype(data(x))>(data(x)));
}

template<class R, class T, class>
convert_t<R,T> digamma(const T& x) {
  prefetch(x);
  return transform(x, digamma_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> exp(const T& x) {
  prefetch(x);
  return transform(x, exp_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> expm1(const T& x) {
  prefetch(x);
  return transform(x, expm1_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> floor(const T& x) {
  prefetch(x);
  return transform(x, floor_functor<R>());
}

template<class T, class>
Array<T,2> inv(const Array<T,2>& A) {
  assert(rows(A) == columns(A));
  Array<T,2> LU(A);
  Array<T,2> B(shape(A));

  /* write identity matrix into B */
  CUDA_CHECK(cudaMemset2DAsync(data(B), stride(B)*sizeof(T), 0,
      rows(B)*sizeof(T), columns(B), stream));
  CUBLAS_CHECK(cublas<T>::copy(cublasHandle, rows(B), scalar<T>::one, 0,
      data(B), stride(B) + 1));

  /* invert via LU factorization with partial pivoting */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, rows(LU), columns(LU), cusolver<T>::CUDA_R, data(LU),
      stride(LU), cusolver<T>::CUDA_R, &bufferOnDeviceBytes,
      &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*std::min(rows(LU),
      columns(LU)));

  CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams,
      rows(LU), columns(LU), cusolver<T>::CUDA_R, data(LU), stride(LU), ipiv,
      cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
      bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_OP_N, rows(B), columns(B), cusolver<T>::CUDA_R, data(LU),
      stride(LU), ipiv, cusolver<T>::CUDA_R, data(B), stride(B), info));

  device_free(ipiv);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  return B;
}

template<class T, class>
Array<T,0> lcholdet(const Array<T,2>& S) {
  Array<T,2> L(S);
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R,
      data(L), stride(L), cusolver<T>::CUDA_R, &bufferOnDeviceBytes,
      &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));

  /* log-determinant is twice the sum of logarithms of elements on the main
   * diagonal, all of which should be positive */
  ///@todo Avoid temporary
  Array<T,0> ldet = sum(transform(L.diagonal(), log_square_functor<T>()));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  return ldet;
}

template<class T, class>
Array<T,0> ldet(const Array<T,2>& A) {
  Array<T,2> LU(A);

  /* LU factorization with partial pivoting */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, rows(LU), columns(LU), cusolver<T>::CUDA_R, data(LU),
      stride(LU), cusolver<T>::CUDA_R, &bufferOnDeviceBytes,
      &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*rows(LU));

  CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams,
      rows(LU), columns(LU), cusolver<T>::CUDA_R, data(LU), stride(LU),
      ipiv, cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));

  /* the LU factorization is with partial pivoting, which means $|A| = (-1)^p
   * |L||U|$, where $p$ is the number of row exchanges in `ipiv`; however,
   * we're taking the logarithm of its absolute value, so can ignore the first
   * term, and the second term is just 1 as $L$ has a unit diagonal; just need
   * $|U|$ here; the logarithm of its absolute value is just the sum of the
   * logarithms of the absolute values of elements on the main diagonal */
  ///@todo Avoid temporary
  Array<T,0> ldet = sum(transform(LU.diagonal(), log_abs_functor<T>()));

  device_free(ipiv);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  return ldet;
}

template<class R, class T, class>
convert_t<R,T> lfact(const T& x) {
  prefetch(x);
  return transform(x, lfact_functor<R>());
}

template<class G, class T, class>
promote_t<G,T> lfact_grad(const G& g, const T& x) {
  prefetch(x);
  return transform_grad(g, x, lfact_grad_functor<value_t<promote_t<G,T>>>());
}

template<class R, class T, class>
convert_t<R,T> lgamma(const T& x) {
  prefetch(x);
  return transform(x, lgamma_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> log(const T& x) {
  prefetch(x);
  return transform(x, log_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> log1p(const T& x) {
  prefetch(x);
  return transform(x, log1p_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> rcp(const T& x) {
  prefetch(x);
  return transform(x, rcp_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> rectify(const T& x) {
  prefetch(x);
  return transform(x, rectify_functor<R>());
}

template<class G, class T, class>
promote_t<G,T> rectify_grad(const G& g, const T& x) {
  prefetch(x);
  return transform_grad(g, x, rectify_grad_functor<
      value_t<promote_t<G,T>>>());
}

template<class R, class T, class>
convert_t<R,T> round(const T& x) {
  prefetch(x);
  return transform(x, round_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> sin(const T& x) {
  prefetch(x);
  return transform(x, sin_functor<R>());
}

template<class R, class T, class>
Array<R,1> single(const T& i, const int n) {
  return for_each(n, single_functor<R,decltype(data(i))>(data(i)));
}

template<class R, class T, class U, class>
Array<R,2> single(const T& i, const U& j, const int m, const int n) {
  return for_each(m, n, single_functor<R,decltype(data(i)),decltype(data(j))>(
      data(i), data(j)));
}

template<class R, class T, class>
convert_t<R,T> sinh(const T& x) {
  prefetch(x);
  return transform(x, sinh_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> sqrt(const T& x) {
  prefetch(x);
  return transform(x, sqrt_functor<R>());
}

template<class R, class T, class>
Array<R,0> sum(const T& x) {
  prefetch(x);
  Array<R,0> z;
  auto y = make_cub(x);
  void* tmp = nullptr;
  size_t bytes = 0;

  CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, y, data(z), size(x),
      stream));
  tmp = device_malloc(bytes);
  CUDA_CHECK(cub::DeviceReduce::Sum(tmp, bytes, y, data(z), size(x),
      stream));
  device_free(tmp);
  return z;
}

template<class R, class T, class>
convert_t<R,T> tan(const T& x) {
  prefetch(x);
  return transform(x, tan_functor<R>());
}

template<class R, class T, class>
convert_t<R,T> tanh(const T& x) {
  prefetch(x);
  return transform(x, tanh_functor<R>());
}

template<class T, class>
Array<T,0> trace(const Array<T,2>& A) {
  assert(rows(A) == columns(A));
  return sum(A.diagonal());
}

template<class T, class>
Array<T,2> transpose(const Array<T,2>& A) {
  prefetch(A);
  Array<T,2> B(make_shape(columns(A), rows(A)));

  dim3 block;
  block.x = CUDA_TRANSPOSE_SIZE;
  block.y = CUDA_TRANSPOSE_SIZE;
  block.z = 1;

  dim3 grid;
  grid.x = (rows(B) + CUDA_TRANSPOSE_SIZE - 1)/CUDA_TRANSPOSE_SIZE;
  grid.y = (columns(B) + CUDA_TRANSPOSE_SIZE - 1)/CUDA_TRANSPOSE_SIZE;
  grid.z = 1;

  size_t shared = CUDA_TRANSPOSE_SIZE*CUDA_TRANSPOSE_SIZE*sizeof(T);

  kernel_transpose<<<grid,block,shared,stream>>>(rows(B), columns(B), data(A),
      stride(A), data(B), stride(B));
  return B;
}

template<class T, class>
Array<T,1> cholmul(const Array<T,2>& S, const Array<T,1>& x) {
  assert(rows(S) == columns(S));
  assert(columns(S) == length(x));
  Array<T,2> L(S);
  Array<T,1> y(x);

  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R,
      data(L), stride(L), cusolver<T>::CUDA_R, &bufferOnDeviceBytes,
      &bufferOnHostBytes));
  void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost,
      bufferOnHostBytes, info));
  CUBLAS_CHECK(cublas<T>::trmv(cublasHandle, CUBLAS_FILL_MODE_LOWER,
      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, length(y), data(L), stride(L),
      data(y), stride(y)));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  return y;
}

template<class T, class>
Array<T,2> cholmul(const Array<T,2>& S, const Array<T,2>& B) {
  assert(rows(S) == columns(S));
  assert(columns(S) == rows(B));
  Array<T,2> L(S);
  Array<T,2> C(B.shape());

  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R,
      data(L), stride(L), cusolver<T>::CUDA_R, &bufferOnDeviceBytes,
      &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));
  CUBLAS_CHECK(cublas<T>::trmm(cublasHandle, CUBLAS_SIDE_LEFT,
      CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rows(C),
      columns(C), scalar<T>::one, data(L), stride(L), data(B), stride(B),
      data(C), stride(C)));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  return C;
}

template<class T, class>
Array<T,2> cholouter(const Array<T,2>& A, const Array<T,2>& S) {
  assert(columns(A) == columns(S));
  assert(rows(S) == columns(S));
  Array<T,2> L(S);
  Array<T,2> C(A.shape());

  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R,
      data(L), stride(L), cusolver<T>::CUDA_R, &bufferOnDeviceBytes,
      &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));
  CUBLAS_CHECK(cublas<T>::trmm(cublasHandle, CUBLAS_SIDE_RIGHT,
      CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rows(C),
      columns(C), scalar<T>::one, data(L), stride(L), data(A), stride(A),
      data(C), stride(C)));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  return C;
}

template<class T, class>
Array<T,1> cholsolve(const Array<T,2>& S, const Array<T,1>& y) {
  assert(rows(S) == columns(S));
  assert(columns(S) == length(y));
  Array<T,2> L(S);
  Array<T,1> x(y);

  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, length(x),
      cusolver<T>::CUDA_R, data(L), stride(L), cusolver<T>::CUDA_R,
      &bufferOnDeviceBytes, &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, length(x), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, length(x), 1, cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, data(x), length(x), info));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  return x;
}

template<class T, class>
Array<T,2> cholsolve(const Array<T,2>& S, const Array<T,2>& C) {
  assert(rows(S) == columns(S));
  assert(columns(S) == rows(C));
  Array<T,2> L(S);
  Array<T,2> B(C);

  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R,
      data(L), stride(L), cusolver<T>::CUDA_R, &bufferOnDeviceBytes,
      &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(B), columns(B), cusolver<T>::CUDA_R,
      data(L), stride(L), cusolver<T>::CUDA_R, data(B), stride(B), info));

  free(bufferOnHost);
  device_free(bufferOnDevice);
  return B;
}

template<class R, class T, class U, class>
convert_t<R,T,U> copysign(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, copysign_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> digamma(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, digamma_functor<R>());
}

template<class T, class>
Array<T,0> dot(const Array<T,1>& x, const Array<T,1>& y) {
  assert(length(x) == length(y));
  prefetch(x);
  prefetch(y);
  Array<T,0> z;
  CUBLAS_CHECK(cublas<T>::dot(cublasHandle, length(x), data(x), stride(x),
      data(y), stride(y), data(z)));
  return z;
}

template<class T, class>
Array<T,0> frobenius(const Array<T,2>& x, const Array<T,2>& y) {
  ///@todo Avoid temporary
  return sum(hadamard(x, y));
}

template<class R, class T, class U, class>
convert_t<R,T,U> gamma_p(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, gamma_p_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> gamma_q(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, gamma_q_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> hadamard(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, multiply_functor<R>());
}

template<class R, class T, class U, class V, class>
convert_t<R,T,U,V> ibeta(const T& x, const U& y, const V& z) {
  prefetch(x);
  prefetch(y);
  prefetch(z);
  return transform(x, y, z, ibeta_functor<R>());
}

template<class T, class>
Array<T,1> inner(const Array<T,2>& A, const Array<T,1>& x) {
  assert(rows(A) == length(x));
  prefetch(A);
  prefetch(x);
  Array<T,1> y(make_shape(rows(A)));
  CUBLAS_CHECK(cublas<T>::gemv(cublasHandle, CUBLAS_OP_T, rows(A), columns(A),
      scalar<T>::one, data(A), stride(A), data(x), stride(x), scalar<T>::zero,
      data(y), stride(y)));
  return y;
}

template<class T, class>
Array<T,2> inner(const Array<T,2>& A, const Array<T,2>& B) {
  assert(rows(A) == rows(B));
  prefetch(A);
  prefetch(B);
  Array<T,2> C(make_shape(rows(A), columns(B)));
  CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
      rows(C), columns(C), rows(A), scalar<T>::one, data(A), stride(A),
      data(B), stride(B), scalar<T>::zero, data(C), stride(C)));
  return C;
}

template<class R, class T, class U, class>
convert_t<R,T,U> lbeta(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lbeta_functor<R>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> lchoose(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lchoose_functor<R>());
}

template<class G, class T, class U, class>
std::pair<promote_t<G,T,U>,promote_t<G,T,U>> lchoose_grad(const G& g,
    const T& x, const U& y) {
  prefetch(g);
  prefetch(x);
  prefetch(y);
  return transform_grad(g, x, y,
      lchoose_grad_functor<value_t<promote_t<G,T,U>>>());
}

template<class R, class T, class U, class>
convert_t<R,T,U> lgamma(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, lgamma_functor<R>());
}

template<class T, class>
Array<T,2> outer(const Array<T,1>& x, const Array<T,1>& y) {
  prefetch(x);
  prefetch(y);
  Array<T,2> A(make_shape(length(x), length(y)));

  /* here, the two vectors are interpreted as single-row matrices, so that the
   * stride between elements becomes the stride between columns; to create the
   * outer product, the first matrix is transposed to a single-column matrix,
   * while the second is not */
  CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
      rows(A), columns(A), 1, scalar<T>::one, data(x), stride(x), data(y),
      stride(y), scalar<T>::zero, data(A), stride(A)));
  return A;
}

template<class T, class>
Array<T,2> outer(const Array<T,2>& A, const Array<T,2>& B) {
  assert(rows(A) == rows(B));
  prefetch(A);
  prefetch(B);
  Array<T,2> C(make_shape(rows(A), rows(B)));
  CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
      rows(C), columns(C), rows(A), scalar<T>::one, data(A), stride(A),
      data(B), stride(B), scalar<T>::zero, data(C), stride(C)));
  return C;
}

template<class R, class T, class U, class>
convert_t<R,T,U> pow(const T& x, const U& y) {
  prefetch(x);
  prefetch(y);
  return transform(x, y, pow_functor<R>());
}

template<class T, class>
Array<T,1> solve(const Array<T,2>& A, const Array<T,1>& y) {
  assert(rows(A) == columns(A));
  assert(columns(A) == length(y));
  Array<T,2> LU(A);
  Array<T,1> x(y);

  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, rows(LU), columns(LU), cusolver<T>::CUDA_R, data(LU),
      stride(LU), cusolver<T>::CUDA_R, &bufferOnDeviceBytes,
      &bufferOnHostBytes));
  void *bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void *bufferOnHost = malloc(bufferOnHostBytes);
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*std::max(1, rows(LU)));

  /* solve via LU factorization with partial pivoting */
  CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams,
      rows(LU), columns(LU), cusolver<T>::CUDA_R, data(LU), stride(LU), ipiv,
      cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
      bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_OP_N, length(x), 1, cusolver<T>::CUDA_R, data(LU), stride(LU),
      ipiv, cusolver<T>::CUDA_R, data(x), length(x), info));

  device_free(ipiv);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  return x;
}

template<class T, class>
Array<T,2> solve(const Array<T,2>& A, const Array<T,2>& C) {
  assert(rows(A) == columns(A));
  assert(columns(A) == rows(C));
  Array<T,2> LU(A);
  Array<T,2> B(C);

  /* solve via LU factorization with partial pivoting */
  size_t bufferOnDeviceBytes = 0, bufferOnHostBytes = 0;
  CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(cusolverDnHandle,
      cusolverDnParams, rows(LU), columns(LU), cusolver<T>::CUDA_R, data(LU),
      stride(LU), cusolver<T>::CUDA_R, &bufferOnDeviceBytes,
      &bufferOnHostBytes));
  void* bufferOnDevice = device_malloc(bufferOnDeviceBytes);
  void* bufferOnHost = malloc(bufferOnHostBytes);
  auto ipiv = (int64_t*)device_malloc(sizeof(int64_t)*std::min(rows(LU),
      columns(LU)));

  CUSOLVER_CHECK_INFO(cusolverDnXgetrf(cusolverDnHandle, cusolverDnParams,
      rows(LU), columns(LU), cusolver<T>::CUDA_R, data(LU), stride(LU), ipiv,
      cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes, bufferOnHost,
      bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXgetrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_OP_N, rows(B), columns(B), cusolver<T>::CUDA_R, data(LU),
      stride(LU), ipiv, cusolver<T>::CUDA_R, data(B), stride(B), info));

  device_free(ipiv);
  free(bufferOnHost);
  device_free(bufferOnDevice);
  return B;
}

}
