/**
 * @file
 * 
 * CUDA boilerplate.
 */
#pragma once

#include "numbirch/array.hpp"
#include "numbirch/type.hpp"
#include "numbirch/macro.hpp"

#include <cassert>

/*
 * If true, all CUDA calls are synchronous, which can be helpful to determine
 * precisely which call causes an error.
 */
#define CUDA_SYNC 0

/*
 * Call a cuda* function and assert success.
 */
#define CUDA_CHECK(call) \
    { \
      cudaError_t err = call; \
      assert(err == cudaSuccess); \
      if (CUDA_SYNC) { \
        cudaError_t err = cudaStreamSynchronize(stream); \
        assert(err == cudaSuccess); \
      } \
    }

namespace numbirch {

extern thread_local int device;
extern thread_local cudaStream_t stream;

/*
 * Preferred thread block size for CUDA kernels.
 */
static const int CUDA_PREFERRED_BLOCK_SIZE = 256;

/*
 * Tile size (number of rows, number of columns) for transpose().
 */
static const int CUDA_TRANSPOSE_SIZE = 16;

/*
 * Initialize CUDA integrations. This should be called during init() by the
 * backend.
 */
void cuda_init();

/*
 * Terminate CUDA integrations. This should be called during term() by the
 * backend.
 */
void cuda_term();

/*
 * Configure thread block size for a vector transformation.
 */
inline dim3 make_block(const int n) {
  dim3 block{0, 0, 0};
  block.x = std::min(n, CUDA_PREFERRED_BLOCK_SIZE);
  if (block.x > 0) {
    block.y = 1;
    block.z = 1;
  }
  return block;
}

/*
 * Configure thread block size for a matrix transformation.
 */
inline dim3 make_block(const int m, const int n) {
  dim3 block{0, 0, 0};
  block.x = std::min(m, CUDA_PREFERRED_BLOCK_SIZE);
  if (block.x > 0) {
    block.y = std::min(n, CUDA_PREFERRED_BLOCK_SIZE/(int)block.x);
    block.z = 1;
  }
  return block;
}

/*
 * Configure grid size for a vector transformation.
 */
inline dim3 make_grid(const int n) {
  dim3 block = make_block(n);
  dim3 grid{0, 0, 0};
  if (block.x > 0) {
    grid.x = (n + block.x - 1)/block.x;
    grid.y = 1;
    grid.z = 1;
  }
  return grid;
}

/*
 * Configure grid size for a matrix transformation.
 */
inline dim3 make_grid(const int m, const int n) {
  dim3 block = make_block(m, n);
  dim3 grid{0, 0, 0};
  if (block.x > 0 && block.y > 0) {
    grid.x = (n + block.x - 1)/block.x;
    grid.y = (m + block.y - 1)/block.y;
    grid.z = 1;
  }
  return grid;
}

/*
 * Prefetch an array onto device.
 */
template<class T, int D>
void prefetch(const Array<T,D>& x) {
  CUDA_CHECK(cudaMemPrefetchAsync(x.data(), x.volume()*sizeof(T), device,
      stream));
}

/*
 * Prefetch a scalar onto device---null operation.
 */
template<class T, class = std::enable_if_t<is_arithmetic<T>::value,int>>
void prefetch(const T& x) {
  //
}

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
 * Unary transform.
 */
template<class T, class U, class Functor>
__global__ void kernel_transform(const int m, const int n, const T* A,
    const int ldA, U* B, const int ldB, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(B, i, j, ldB) = f(element(A, i, j, ldA));
  }
}
template<class T, class Functor>
auto transform(const T& x, Functor f) {
  using V = decltype(f(value_t<T>()));
  constexpr int D = dimension_v<T>;
  auto y = Array<V,D>(shape(x));
  auto m = rows(x);
  auto n = columns(x);
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, data(x), stride(x), data(y),
      stride(y), f);
  return y;
}

/*
 * Binary transform.
 */
template<class T, class U, class V, class Functor>
__global__ void kernel_transform(const int m, const int n, const T A,
    const int ldA, const U B, const int ldB, V C, const int ldC,
    Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(C, i, j, ldC) = f(element(A, i, j, ldA), element(B, i, j, ldB));
  }
}
template<class T, class U, class Functor>
auto transform(const T& x, const U& y, Functor f) {
  assert(conforms(x, y));
  using V = decltype(f(value_t<T>(),value_t<U>()));
  constexpr int D = dimension_v<T>;
  auto z = Array<V,D>(shape(x));
  auto m = rows(x);
  auto n = columns(x);
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, data(x), stride(x), data(y),
      stride(y), data(z), stride(z), f);
  return z;
}

/*
 * Matrix ternary transform.
 */
template<class T, class U, class V, class W, class Functor>
__global__ void kernel_transform(const int m, const int n, const T* A,
    const int ldA, const U* B, const int ldB, const V* C, const int ldC,
    W* D, const int ldD, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(D, i, j, ldD) = f(element(A, i, j, ldA), element(B, i, j, ldB),
        element(C, i, j, ldC));
  }
}
template<class T, class U, class V, class W, class Functor>
void transform(const int m, const int n, const T* A, const int ldA,
    const U* B, const int ldB, const V* C, const int ldC, W* D, const int ldD,
    Functor f) {
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, A, ldA, B, ldB, C, ldC, D,
      ldD, f);
}

/*
 * Matrix ternary transform with two outputs.
 */
template<class T, class U, class V, class W, class X, class Functor>
__global__ void kernel_transform(const int m, const int n, const T* A,
    const int ldA, const U* B, const int ldB, const V* C, const int ldC,
    W* D, const int ldD, X* E, const int ldE, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    auto pair = f(element(A, i, j, ldA), element(B, i, j, ldB),
        element(C, i, j, ldC));
    element(D, i, j, ldD) = pair.first;
    element(E, i, j, ldE) = pair.second;
  }
}
template<class T, class U, class V, class W, class X, class Functor>
void transform(const int m, const int n, const T* A, const int ldA,
    const U* B, const int ldB, const V* C, const int ldC, W* D, const int ldD,
    X* E, const int ldE, Functor f) {
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, A, ldA, B, ldB, C, ldC, D,
      ldD, E, ldE, f);
}

/*
 * Matrix quaternary transform.
 */
template<class T, class U, class V, class W, class X, class Functor>
__global__ void kernel_transform(const int m, const int n, const T* A,
    const int ldA, const U* B, const int ldB, const V* C, const int ldC,
    const W* D, const int ldD, X* E, const int ldE, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(E, i, j, ldE) = f(element(A, i, j, ldA), element(B, i, j, ldB),
        element(C, i, j, ldC), element(D, i, j, ldD));
  }
}
template<class T, class U, class V, class W, class X, class Functor>
void transform(const int m, const int n, const T* A, const int ldA,
    const U* B, const int ldB, const V* C, const int ldC, const W* D,
    const int ldD, X* E, const int ldE, Functor f) {
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, A, ldA, B, ldB, C, ldC, D,
      ldD, E, ldE, f);
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

}
