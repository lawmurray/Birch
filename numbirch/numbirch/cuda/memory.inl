/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/cuda/cuda.hpp"

namespace numbirch {
  
template<class T>
__global__ void kernel_memset(void* dst, const size_t dpitch, const T value,
    const size_t width, const size_t height) {
  auto A = (T*)dst;
  auto ld = dpitch/sizeof(T);
  auto m = width/sizeof(T);
  auto n = height/sizeof(T);
  for (auto j = blockIdx.y*blockDim.y + threadIdx.y; j < n;
      j += gridDim.y*blockDim.y) {
    for (auto i = blockIdx.x*blockDim.x + threadIdx.x; i < m;
        i += gridDim.x*blockDim.x) {
      A[i + j*ld] = value;
    }
  }
}

template<class T, class>
void memset(void* dst, const size_t dpitch, const T value, const size_t width,
    const size_t height) {
  if (width > 0 && height > 0) {
    auto grid = make_grid(width/sizeof(T), height);
    auto block = make_block(width/sizeof(T), height);
    CUDA_LAUNCH(kernel_memset<<<grid,block,0,stream>>>(dst, dpitch, value,
        width, height));
  }
}

}
