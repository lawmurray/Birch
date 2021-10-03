/**
 * @file
 * 
 * CUDA boilerplate.
 */
#pragma once

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
 * Matrix for-each.
 */
template<class T, class Functor>
__global__ void kernel_for_each(const int m, const int n, T* A, const int ldA,
    Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    A[i + j*ldA] = f(i, j);
  }
}
template<class T, class Functor>
void for_each(const int m, const int n, T* A, const int ldA, Functor f) {
  auto grid = make_grid(m, n);
  auto block = make_block(m, n);
  kernel_for_each<<<grid,block,0,stream>>>(m, n, A, ldA, f);
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
 * Matrix quaternary transform.
 */
template<class T, class Functor>
__global__ void kernel_transform(const int m, const int n, const T* A,
    const int ldA, const T* B, const int ldB, const T* C, const int ldC,
    const T* D, const int ldD, T* E, const int ldE, Functor f) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    E[i + j*ldE] = f(A[i + j*ldA], B[i + j*ldB], C[i + j*ldC], D[i + j*ldD]);
  }
}
template<class T, class Functor>
void transform(const int m, const int n, const T* A,
    const int ldA, const T* B, const int ldB, const T* C, const int ldC,
    const T* D, const int ldD, T* E, const int ldE, Functor f) {
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
    B[i + j*ldB] = tile[threadIdx.y][threadIdx.x];
  }
}

}
