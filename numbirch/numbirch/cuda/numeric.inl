/**
 * @file
 */
#pragma once

#include "numbirch/utility.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/cub.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/numeric.hpp"
#include "numbirch/transform.hpp"
#include "numbirch/reduce.hpp"
#include "numbirch/memory.hpp"

#include "numbirch/common/transform.inl"
#include "numbirch/cuda/transform.inl"

namespace numbirch {
/*
 * Tile size (number of rows, number of columns) for transpose().
 */
static const int CUDA_TRANSPOSE_SIZE = 16;

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
    tile[threadIdx.x][threadIdx.y] = get(A, i, j, ldA);
  }
  __syncthreads();
  i = blockIdx.x*blockDim.x + threadIdx.x;
  j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    get(B, i, j, ldB) = tile[threadIdx.y][threadIdx.x];
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
Array<T,2> chol(const Array<T,2>& S) {
  // assert(rows(S) == columns(S));
  // Array<T,2> L(shape(S));
  // auto S1 = make_eigen(S);
  // auto L1 = make_eigen(L);
  // auto llt = S1.llt();
  // if (llt.info() == Eigen::Success) {
  //   L1 = llt.matrixL();
  // } else {
  //   L.fill(T(0.0/0.0));
  // }
  // return L;
}

template<class T, class>
Array<T,2> chol_grad(const Array<T,2>& g, const Array<T,2>& L,
    const Array<T,2>& S) {
  // assert(rows(g) == columns(g));
  // assert(rows(L) == columns(L));
  // assert(rows(S) == columns(S));
  // assert(rows(g) == rows(L));
  // assert(rows(g) == rows(S));
  // Array<T,2> gS(shape(S));
  // auto g1 = make_eigen(g);
  // auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  // auto S1 = make_eigen(S);
  // auto gS1 = make_eigen(gS);
  // gS1 = (U1*g1).template triangularView<Eigen::Lower>();
  // gS1.diagonal() *= 0.5;
  // gS1 = U1.solve(U1.solve(gS1).transpose());
  // gS1 = (gS1.transpose() + gS1).template triangularView<Eigen::Lower>();
  // gS1.diagonal() *= 0.5;
  // return gS;
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
  void* bufferOnHost = host_malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(B), columns(B), cusolver<T>::CUDA_R,
      data(L), stride(L), cusolver<T>::CUDA_R, data(B), stride(B), info));

  host_free(bufferOnHost);
  device_free(bufferOnDevice);
  return B;
}

template<class T, class>
Array<T,2> cholinv_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& L) {
  // assert(rows(L) == columns(L));
  // Array<T,2> gS(shape(L)), gL(shape(L));
  // auto g1 = make_eigen(g);
  // auto B1 = make_eigen(B);
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto gS1 = make_eigen(gS);
  // auto gL1 = make_eigen(gL);
  // gS1.noalias() = -B1.transpose()*g1*B1.transpose();
  // gL1 = ((gS1 + gS1.transpose())*L1).template triangularView<Eigen::Lower>();
  // return gL;
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
  void *bufferOnHost = host_malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, length(x), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, length(x), 1, cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, data(x), length(x), info));

  host_free(bufferOnHost);
  device_free(bufferOnDevice);
  return x;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,1>> cholsolve_grad(const Array<T,1>& g,
    const Array<T,1>& x, const Array<T,2>& L, const Array<T,1>& y) {
  // assert(length(g) == length(y));
  // assert(rows(L) == columns(L));
  // assert(columns(L) == length(y));
  // Array<T,2> gS(shape(L));
  // Array<T,2> gL(shape(L));
  // Array<T,1> gy(shape(y));
  // auto g1 = make_eigen(g);
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  // auto x1 = make_eigen(x);
  // auto y1 = make_eigen(y);
  // auto gS1 = make_eigen(gS);
  // auto gL1 = make_eigen(gL);
  // auto gy1 = make_eigen(gy);
  // gy1.noalias() = U1.solve(L1.solve(g1));
  // gS1.noalias() = -gy1*x1.transpose();
  // gL1 = ((gS1 + gS1.transpose())*L1).template triangularView<Eigen::Lower>();
  // return std::make_pair(gL, gy);
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
  void* bufferOnHost = host_malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));
  CUSOLVER_CHECK_INFO(cusolverDnXpotrs(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(B), columns(B), cusolver<T>::CUDA_R,
      data(L), stride(L), cusolver<T>::CUDA_R, data(B), stride(B), info));

  host_free(bufferOnHost);
  device_free(bufferOnDevice);
  return B;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,2>> cholsolve_grad(const Array<T,2>& g,
    const Array<T,2>& B, const Array<T,2>& L, const Array<T,2>& C) {
  // assert(rows(L) == columns(L));
  // assert(columns(L) == rows(C));
  // Array<T,2> gS(shape(L));
  // Array<T,2> gL(shape(L));
  // Array<T,2> gC(shape(C));
  // auto g1 = make_eigen(g);
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  // auto B1 = make_eigen(B);
  // auto C1 = make_eigen(C);
  // auto gS1 = make_eigen(gS);
  // auto gL1 = make_eigen(gL);
  // auto gC1 = make_eigen(gC);
  // gC1.noalias() = U1.solve(L1.solve(g1));
  // gS1.noalias() = -gC1*B1.transpose();
  // gL1 = ((gS1 + gS1.transpose())*L1).template triangularView<Eigen::Lower>();
  // return std::make_pair(gL, gC);
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
  return sum(mul(x, y));
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
  void* bufferOnHost = host_malloc(bufferOnHostBytes);
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
  host_free(bufferOnHost);
  device_free(bufferOnDevice);
  return B;
}

template<class T, class>
Array<T,2> inv_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& A) {
  // assert(rows(B) == columns(B));
  // assert(rows(A) == columns(A));
  // Array<T,2> gA(shape(A));
  // auto g1 = make_eigen(g);
  // auto B1 = make_eigen(B);
  // auto gA1 = make_eigen(gA);
  // gA1.noalias() = -B1.transpose()*g1*B1.transpose();
  // return gA;
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
  void* bufferOnHost = host_malloc(bufferOnHostBytes);

  CUSOLVER_CHECK_INFO(cusolverDnXpotrf(cusolverDnHandle, cusolverDnParams,
      CUBLAS_FILL_MODE_LOWER, rows(L), cusolver<T>::CUDA_R, data(L),
      stride(L), cusolver<T>::CUDA_R, bufferOnDevice, bufferOnDeviceBytes,
      bufferOnHost, bufferOnHostBytes, info));

  /* log-determinant is twice the sum of logarithms of elements on the main
   * diagonal, all of which should be positive */
  ///@todo Avoid temporary
  Array<T,0> ldet = sum(transform(L.diagonal(), log_square_functor()));

  host_free(bufferOnHost);
  device_free(bufferOnDevice);
  return ldet;
}

template<class T, class>
Array<T,2> lcholdet_grad(const Array<T,0>& g, const Array<T,0>& d,
    const Array<T,2>& L) {
  // Array<T,2> gL(shape(L));
  // auto g1 = make_eigen(g);
  // auto L1 = make_eigen(L);
  // auto gL1 = make_eigen(gL);
  // gL1 = (T(2.0)*g.value()/L1.diagonal().array()).matrix().asDiagonal();
  // assert(gL1.isDiagonal());
  // return gL;
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
  void* bufferOnHost = host_malloc(bufferOnHostBytes);
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
  Array<T,0> ldet = sum(transform(LU.diagonal(), log_abs_functor()));

  device_free(ipiv);
  host_free(bufferOnHost);
  device_free(bufferOnDevice);
  return ldet;
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
      rows(C), columns(C), columns(A), scalar<T>::one, data(A), stride(A),
      data(B), stride(B), scalar<T>::zero, data(C), stride(C)));
  return C;
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

  CUDA_LAUNCH(kernel_transpose<<<grid,block,shared,stream>>>(rows(B),
      columns(B), data(A), stride(A), data(B), stride(B)));
  return B;
}

template<class T, class>
Array<T,1> triinner(const Array<T,2>& L, const Array<T,1>& x) {
  assert(rows(L) == columns(L));
  assert(columns(L) == length(x));
  prefetch(L);
  prefetch(x);
  Array<T,1> y(x);
  CUBLAS_CHECK(cublas<T>::trmv(cublasHandle, CUBLAS_FILL_MODE_LOWER,
      CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, rows(L), data(L), stride(L), data(y),
      stride(y)));
  return y;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,1>> triinner_grad(const Array<T,1>& g,
    const Array<T,1>& y, const Array<T,2>& L, const Array<T,1>& x) {
  // assert(rows(L) == length(x));
  // Array<T,2> gL(shape(L));
  // Array<T,1> gx(shape(x));
  // auto g1 = make_eigen(g);
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto x1 = make_eigen(x);
  // auto gL1 = make_eigen(gL);
  // auto gx1 = make_eigen(gx);
  // gL1 = (x1*g1.transpose()).template triangularView<Eigen::Lower>();
  // gx1.noalias() = L1*g1;
  // return std::make_pair(gL, gx);
}

template<class T, class>
Array<T,2> triinner(const Array<T,2>& L) {
  // return triinner(L, L);
}

template<class T, class>
Array<T,2> triinner_grad(const Array<T,2>& g, const Array<T,2>& C,
    const Array<T,2>& L) {
  // Array<T,2> gL(shape(L));
  // auto g1 = make_eigen(g);
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto gL1 = make_eigen(gL);
  // gL1 = (L1*(g1 + g1.transpose())).template triangularView<Eigen::Lower>();
  // return gL;
}

template<class T, class>
Array<T,2> triinner(const Array<T,2>& L, const Array<T,2>& B) {
  assert(rows(L) == columns(L));
  assert(columns(L) == rows(B));
  prefetch(L);
  prefetch(B);
  Array<T,2> C(make_shape(rows(B), columns(B)));
  CUBLAS_CHECK(cublas<T>::trmm(cublasHandle, CUBLAS_SIDE_LEFT,
      CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, rows(B),
      columns(B), scalar<T>::one, data(L), stride(L), data(B), stride(B),
      data(C), stride(C)));
  return C;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,2>> triinner_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& L, const Array<T,2>& B) {
  // Array<T,2> gL(shape(L));
  // Array<T,2> gB(shape(B));
  // auto g1 = make_eigen(g);
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto B1 = make_eigen(B);
  // auto gL1 = make_eigen(gL);
  // auto gB1 = make_eigen(gB);
  // gL1 = (B1*g1.transpose()).template triangularView<Eigen::Lower>();
  // gB1.noalias() = L1*g1;
  // return std::make_pair(gL, gB);
}

template<class T, class>
Array<T,2> triinv(const Array<T,2>& L) {
  // assert(rows(L) == columns(L));
  // Array<T,2> B(shape(L));
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto B1 = make_eigen(B);
  // B1.noalias() = L1.solve(B1.Identity(rows(B), columns(B)));
  // return B;
}

template<class T, class>
Array<T,2> triinv_grad(const Array<T,2>& g, const Array<T,2>& B,
    const Array<T,2>& L) {
  // assert(rows(B) == columns(B));
  // assert(rows(L) == columns(L));
  // Array<T,2> gL(shape(L));
  // auto g1 = make_eigen(g);
  // auto B1 = make_eigen(B).transpose().template triangularView<Eigen::Upper>();
  // auto gL1 = make_eigen(gL);
  // gL1 = (-(B1*g1*B1)).template triangularView<
  //     Eigen::Lower>();
  // return gL;
}

template<class T, class>
Array<T,1> trimul(const Array<T,2>& L, const Array<T,1>& x) {
  assert(rows(L) == columns(L));
  assert(columns(L) == length(x));
  prefetch(L);
  prefetch(x);
  Array<T,1> y(x);
  CUBLAS_CHECK(cublas<T>::trmv(cublasHandle, CUBLAS_FILL_MODE_LOWER,
      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rows(L), data(L), stride(L), data(y),
      stride(y)));
  return y;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,1>> trimul_grad(const Array<T,1>& g,
    const Array<T,1>& y, const Array<T,2>& L, const Array<T,1>& x) {
  // assert(columns(L) == length(x));
  // Array<T,2> gL(shape(L));
  // Array<T,1> gx(shape(x));
  // auto g1 = make_eigen(g);
  // auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  // auto x1 = make_eigen(x);
  // auto gL1 = make_eigen(gL);
  // auto gx1 = make_eigen(gx);
  // gL1 = (g1*x1.transpose()).template triangularView<Eigen::Lower>();
  // gx1.noalias() = U1*g1;
  // return std::make_pair(gL, gx);
}

template<class T, class>
Array<T,2> trimul(const Array<T,2>& L, const Array<T,2>& B) {
  assert(rows(L) == columns(L));
  assert(columns(L) == rows(B));
  prefetch(L);
  prefetch(B);
  Array<T,2> C(make_shape(rows(B), columns(B)));
  CUBLAS_CHECK(cublas<T>::trmm(cublasHandle, CUBLAS_SIDE_LEFT,
      CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rows(B),
      columns(B), scalar<T>::one, data(L), stride(L), data(B), stride(B),
      data(C), stride(C)));
  return C;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,2>> trimul_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& L, const Array<T,2>& B) {
  // assert(columns(L) == rows(B));
  // Array<T,2> gL(shape(L));
  // Array<T,2> gB(shape(B));
  // auto g1 = make_eigen(g);
  // auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  // auto B1 = make_eigen(B);
  // auto gL1 = make_eigen(gL);
  // auto gB1 = make_eigen(gB);
  // gL1 = (g1*B1.transpose()).template triangularView<Eigen::Lower>();
  // gB1.noalias() = U1*g1;
  // return std::make_pair(gL, gB);
}

template<class T, class>
Array<T,2> triouter(const Array<T,2>& L) {
  // return triouter(L, L);
}

template<class T, class>
Array<T,2> triouter_grad(const Array<T,2>& g, const Array<T,2>& C,
    const Array<T,2>& L) {
  // Array<T,2> gL(shape(L));
  // auto g1 = make_eigen(g);
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto gL1 = make_eigen(gL);
  // gL1 = ((g1 + g1.transpose())*L1).template triangularView<Eigen::Lower>();
  // return gL;
}

template<class T, class>
Array<T,2> triouter(const Array<T,2>& A, const Array<T,2>& L) {
  // assert(columns(A) == columns(L));
  // Array<T,2> C(make_shape(rows(A), rows(L)));
  // auto A1 = make_eigen(A);
  // auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  // auto C1 = make_eigen(C);
  // C1.noalias() = A1*U1;
  // return C;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,2>> triouter_grad(const Array<T,2>& g,
    const Array<T,2>& C, const Array<T,2>& A, const Array<T,2>& L) {
  // Array<T,2> gA(shape(A));
  // Array<T,2> gL(shape(L));
  // auto g1 = make_eigen(g);
  // auto A1 = make_eigen(A);
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto gA1 = make_eigen(gA);
  // auto gL1 = make_eigen(gL);
  // gA1.noalias() = g1*L1;
  // gL1 = (g1.transpose()*A1).template triangularView<Eigen::Lower>();
  // return std::make_pair(gA, gL);
}

template<class T, class>
Array<T,1> trisolve(const Array<T,2>& L, const Array<T,1>& y) {
  // assert(rows(L) == columns(L));
  // assert(columns(L) == length(y));
  // Array<T,1> x(shape(y));
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto x1 = make_eigen(x);
  // auto y1 = make_eigen(y);
  // x1.noalias() = L1.solve(y1);
  // return x;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,1>> trisolve_grad(const Array<T,1>& g,
    const Array<T,1>& x, const Array<T,2>& L, const Array<T,1>& y) {
  // assert(length(g) == length(y));
  // assert(rows(L) == columns(L));
  // assert(columns(L) == length(y));
  // Array<T,2> gL(shape(L));
  // Array<T,1> gy(shape(y));
  // auto g1 = make_eigen(g);
  // auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  // auto x1 = make_eigen(x);
  // auto y1 = make_eigen(y);
  // auto gL1 = make_eigen(gL);
  // auto gy1 = make_eigen(gy);
  // gy1.noalias() = U1.solve(g1);
  // gL1 = (-gy1*x1.transpose()).template triangularView<Eigen::Lower>();
  // return std::make_pair(gL, gy);
}

template<class T, class>
Array<T,2> trisolve(const Array<T,2>& L, const Array<T,2>& C) {
  // assert(rows(L) == columns(L));
  // assert(columns(L) == rows(C));
  // Array<T,2> B(shape(C));
  // auto L1 = make_eigen(L).template triangularView<Eigen::Lower>();
  // auto B1 = make_eigen(B);
  // auto C1 = make_eigen(C);
  // B1.noalias() = L1.solve(C1);
  // return B;
}

template<class T, class>
std::pair<Array<T,2>,Array<T,2>> trisolve_grad(const Array<T,2>& g,
    const Array<T,2>& B, const Array<T,2>& L, const Array<T,2>& C) {
  // assert(rows(g) == rows(L));
  // assert(columns(g) == columns(C));
  // assert(rows(L) == columns(L));
  // assert(columns(L) == rows(C));
  // Array<T,2> gL(shape(L));
  // Array<T,2> gC(shape(C));
  // auto g1 = make_eigen(g);
  // auto U1 = make_eigen(L).transpose().template triangularView<Eigen::Upper>();
  // auto B1 = make_eigen(B);
  // auto C1 = make_eigen(C);
  // auto gL1 = make_eigen(gL);
  // auto gC1 = make_eigen(gC);
  // gC1.noalias() = U1.solve(g1);
  // gL1 = (-gC1*B1.transpose()).template triangularView<Eigen::Lower>();
  // return std::make_pair(gL, gC);
}

}
