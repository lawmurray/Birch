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
template<class T, class R, class Functor>
__global__ void kernel_transform(const int m, const int n, const T A,
    const int ldA, R B, const int ldB, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(B, i, j, ldB) = f(element(A, i, j, ldA));
  }
}
template<class T, class Functor>
auto transform(const T& x, Functor f) {
  using R = decltype(f(value_t<T>()));
  constexpr int D = dimension_v<T>;
  auto y = Array<R,D>(shape(x));
  auto m = rows(x);
  auto n = columns(x);
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, data(x), stride(x), data(y),
      stride(y), f);
  return y;
}

/*
 * Gradient of unary transform.
 */
template<class G, class T, class Functor>
auto transform_grad(const G& g, const T& x, Functor f) {
  return transform(g, x, f);  // same as binary transform
}

/*
 * Binary transform.
 */
template<class T, class U, class R, class Functor>
__global__ void kernel_transform(const int m, const int n, const T A,
    const int ldA, const U B, const int ldB, R C, const int ldC,
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
  using R = decltype(f(value_t<T>(),value_t<U>()));
  constexpr int D = std::max(dimension_v<T>, dimension_v<U>);
  auto m = std::max(rows(x), rows(y));
  auto n = std::max(columns(x), columns(y));
  auto z = Array<R,D>(m, n);
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, data(x), stride(x), data(y),
      stride(y), data(z), stride(z), f);
  return z;
}

/*
 * Gradient of binary transform.
 */
template<class G, class T, class U, class V, class W, class Functor>
__global__ void kernel_transform_grad(const int m, const int n, const G g,
    const int ldg, const T A, const int ldA, const U B, const int ldB,
    V GA, const int ldGA, W GB, const int ldGB, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    auto pair = f(element(g, i, j, ldg), element(A, i, j, ldA),
        element(B, i, j, ldB));
    element(GA, i, j, ldGA) = pair.first;
    element(GB, i, j, ldGB) = pair.second;
  }
}
template<class G, class T, class U, class Functor>
auto transform_grad(const G& g, const T& x, const U& y, Functor f) {
  assert(conforms(x, y));
  using P = decltype(f(value_t<G>(),value_t<T>(),value_t<U>()));
  using V = typename P::first_type;
  using W = typename P::second_type;
  constexpr int D = std::max(std::max(dimension_v<G>, dimension_v<T>),
      dimension_v<U>);
  auto m = std::max(std::max(rows(g), rows(x)), rows(y));
  auto n = std::max(std::max(columns(g), columns(x)), columns(y));
  auto a = Array<V,D>(m, n);
  auto b = Array<W,D>(m, n);
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform_grad<<<grid,block,0,stream>>>(m, n, data(g), stride(g),
      data(x), stride(x), data(y), stride(y), data(a), stride(a), data(b),
      stride(b), f);
  return std::make_pair(a, b);
}

/*
 * Ternary transform.
 */
template<class T, class U, class V, class R, class Functor>
__global__ void kernel_transform(const int m, const int n, const T A,
    const int ldA, const U B, const int ldB, const V C, const int ldC,
    R D, const int ldD, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    element(D, i, j, ldD) = f(element(A, i, j, ldA), element(B, i, j, ldB),
        element(C, i, j, ldC));
  }
}
template<class T, class U, class V, class Functor>
auto transform(const T& x, const U& y, const V& z, Functor f) {
  assert(conforms(x, y) && conforms(y, z));
  using R = decltype(f(value_t<T>(),value_t<U>(),value_t<V>()));
  constexpr int D = std::max(std::max(dimension_v<T>, dimension_v<U>),
      dimension_v<V>);
  auto m = std::max(std::max(rows(x), rows(y)), rows(z));
  auto n = std::max(std::max(columns(x), columns(y)), columns(z));
  auto a = Array<R,D>(m, n);
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_transform<<<grid,block,0,stream>>>(m, n, data(x), stride(x), data(y),
      stride(y), data(z), stride(z), data(a), stride(a), f);
  return a;
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
