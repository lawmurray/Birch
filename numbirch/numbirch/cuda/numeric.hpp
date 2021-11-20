/**
 * @file
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/cub.hpp"
#include "numbirch/cuda/transform.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/common/functor.hpp"
#include "numbirch/binary.hpp"
#include "numbirch/memory.hpp"

namespace numbirch {
/*
 * Tile size (number of rows, number of columns) for transpose().
 */
static const int CUDA_TRANSPOSE_SIZE = 16;

/*
 * Matrix for-each.
 */
template<class T, class Functor>
__global__ void kernel_for_each(const int m, const int n, T* A, const int ldA,
    Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(A, i, j, ldA) = f(i, j);
  }
}
template<class Functor>
auto for_each(const int m, const int n, Functor f) {
  auto A = Array<decltype(f(0,0)),2>(make_shape(m, n));
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_for_each<<<grid,block,0,stream>>>(m, n, data(A), stride(A), f);
  return A;
}
template<class Functor>
auto for_each(const int n, Functor f) {
  auto x = Array<decltype(f(0,0)),1>(make_shape(n));
  auto grid = make_grid(n, 1);
  auto block = make_block(n, 1);
  kernel_for_each<<<grid,block,0,stream>>>(n, 1, data(x), stride(x), f);
  return x;
}

/*
 * Matrix transpose kernel.
 */
template<class T>
__global__ void kernel_transpose(const int m, const int n, const T* A,
    const int ldA, T* B, const int ldB) {
  __shared__ T tile[CUDA_TRANSPOSE_SIZE][CUDA_TRANSPOSE_SIZE + 1];
  // ^ +1 reduce shared memory bank conflicts

  auto i = blockIdx.y*blockDim.y + threadIdx.x;
  auto j = blockIdx.x*blockDim.x + threadIdx.y;
  if (i < n && j < m) {
    tile[threadIdx.x][threadIdx.y] = element(A, i, j, ldA);
  }
  __syncthreads();
  i = blockIdx.x*blockDim.x + threadIdx.x;
  j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(B, i, j, ldB) = tile[threadIdx.y][threadIdx.x];
  }
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
Array<R,0> count(const T& x) {
  ///@todo Avoid temporary
  return sum(transform(x, count_functor<R>()));
}

template<class R, class T, class>
Array<R,2> diagonal(const T& x, const int n) {
  return for_each(n, n, diagonal_functor<R,decltype(data(x))>(data(x)));
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
Array<R,1> single(const T& i, const int n) {
  return for_each(n, single_functor<R,decltype(data(i))>(data(i)));
}

template<class R, class T, class U, class>
Array<R,2> single(const T& i, const U& j, const int m, const int n) {
  return for_each(m, n, single_functor<R,decltype(data(i)),decltype(data(j))>(
      data(i), data(j)));
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
  Array<T,2> C(shape(B));

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
  Array<T,2> C(shape(A));

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

template<class T, class>
Array<T,1> inner(const Array<T,2>& A, const Array<T,1>& x) {
  assert(rows(A) == length(x));
  prefetch(A);
  prefetch(x);
  Array<T,1> y(make_shape(columns(A)));
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
  Array<T,2> C(make_shape(columns(A), columns(B)));
  CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
      rows(C), columns(C), rows(A), scalar<T>::one, data(A), stride(A),
      data(B), stride(B), scalar<T>::zero, data(C), stride(C)));
  return C;
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
  assert(columns(A) == columns(B));
  prefetch(A);
  prefetch(B);
  Array<T,2> C(make_shape(rows(A), rows(B)));
  CUBLAS_CHECK(cublas<T>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
      rows(C), columns(C), rows(A), scalar<T>::one, data(A), stride(A),
      data(B), stride(B), scalar<T>::zero, data(C), stride(C)));
  return C;
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
