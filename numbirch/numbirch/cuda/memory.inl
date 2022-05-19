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
  CUDA_LAUNCH(kernel_memset<<<grid,block,0,stream>>>(dst, dpitch, value,
      width, height));
}

}
