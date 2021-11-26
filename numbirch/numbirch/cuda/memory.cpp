/**
 * @file
 */
#include "numbirch/memory.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"

namespace numbirch {

void* extent_alloc(extent_hooks_t *extent_hooks, void *new_addr, size_t size,
    size_t alignment, bool *zero, bool *commit, unsigned arena_ind) {
  if (!new_addr) {
    CUDA_CHECK(cudaMallocManaged(&new_addr, size));
    *commit = true;
  }
  if (*zero) {
    CUDA_CHECK(cudaMemset(new_addr, 0, size));
  }
  return new_addr;
}

bool extent_dalloc(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFree(addr));
  return false;
}

void extent_destroy(extent_hooks_t *extent_hooks, void *addr, size_t size,
    bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFree(addr));
}

void* device_extent_alloc(extent_hooks_t *extent_hooks, void *new_addr,
    size_t size, size_t alignment, bool *zero, bool *commit,
    unsigned arena_ind) {
  if (!new_addr) {
    CUDA_CHECK(cudaMallocManaged(&new_addr, size));
    *commit = true;
  }
  if (*zero) {
    CUDA_CHECK(cudaMemset(new_addr, 0, size));
  }
  return new_addr;
}

bool device_extent_dalloc(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFree(addr));
  return false;
}

void device_extent_destroy(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFree(addr));
}

void* host_extent_alloc(extent_hooks_t *extent_hooks, void *new_addr,
    size_t size, size_t alignment, bool *zero, bool *commit,
    unsigned arena_ind) {
  if (!new_addr) {
    CUDA_CHECK(cudaMallocHost(&new_addr, size));
    *commit = true;
  }
  if (*zero) {
    CUDA_CHECK(cudaMemset(new_addr, 0, size));
  }
  return new_addr;
}

bool host_extent_dalloc(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFreeHost(addr));
  return false;
}

void host_extent_destroy(extent_hooks_t *extent_hooks, void *addr,
    size_t size, bool committed, unsigned arena_ind) {
  CUDA_CHECK(cudaFreeHost(addr));
}

void init() {
  cuda_init();
  cublas_init();
  cusolver_init();
  jemalloc_init();
}

void wait() {
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void term() {
  jemalloc_term();
  cusolver_term();
  cublas_term();
  cuda_term();
}

void memcpy(void* dst, const size_t dpitch, const void* src,
    const size_t spitch, const size_t width, const size_t height) {
  CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
      cudaMemcpyDefault, stream));
}

template<class T>
__global__ void kernel_memset(void* dst, const size_t dpitch, const T value,
    const size_t width, const size_t height) {
  auto i = blockIdx.x*blockDim.x + threadIdx.x;
  auto j = blockIdx.y*blockDim.y + threadIdx.y;
  auto A = (T*)dst;

  if (i < width/sizeof(T) && j < height) {
    A[i + j*dpitch/sizeof(T)] = value;
  }
}

template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int>>
void memset(void* dst, const size_t dpitch, const T value, const size_t width,
    const size_t height) {
  auto grid = make_grid(width/sizeof(T), height);
  auto block = make_block(width/sizeof(T), height);
  kernel_memset<<<grid,block,0,stream>>>(dst, dpitch, value, width, height);
}

template void memset(void*, const size_t, const double, const size_t,
    const size_t);
template void memset(void*, const size_t, const float, const size_t,
    const size_t);
template void memset(void*, const size_t, const int, const size_t,
    const size_t);
template void memset(void*, const size_t, const bool, const size_t,
    const size_t);
}
