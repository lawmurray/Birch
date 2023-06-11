/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/utility.hpp"
#include "numbirch/cuda/cuda.hpp"

namespace numbirch {

template<class T, class U>
__global__ void kernel_memcpy(T* dst, const int dpitch, const U* src,
    const int spitch, const int width, const int height) {
  for (auto j = blockIdx.y*blockDim.y + threadIdx.y; j < height;
      j += gridDim.y*blockDim.y) {
    for (auto i = blockIdx.x*blockDim.x + threadIdx.x; i < width;
        i += gridDim.x*blockDim.x) {
      get(dst, i, j, dpitch) = get(src, i, j, spitch);
    }
  }
}

template<class T, class U>
__global__ void kernel_memset(T* dst, const int dpitch, const U value,
    const int width, const int height) {
  for (auto j = blockIdx.y*blockDim.y + threadIdx.y; j < height;
      j += gridDim.y*blockDim.y) {
    for (auto i = blockIdx.x*blockDim.x + threadIdx.x; i < width;
        i += gridDim.x*blockDim.x) {
      get(dst, i, j, dpitch) = get(value);
    }
  }
}

template<arithmetic T, arithmetic U>
void memcpy(T* dst, const int dpitch, const U* src, const int spitch,
    const int width, const int height) {
  if (width > 0 && height > 0) {
    auto grid = make_grid(width, height);
    auto block = make_block(width, height);
    CUDA_LAUNCH(kernel_memcpy<<<grid,block,0,stream>>>(dst, dpitch, src,
        spitch, width, height));
  }
}

template<arithmetic T, arithmetic U>
void memset(T* dst, const int dpitch, const U value, const int width,
    const int height) {
  if (width > 0 && height > 0) {
    auto grid = make_grid(width, height);
    auto block = make_block(width, height);
    CUDA_LAUNCH(kernel_memset<<<grid,block,0,stream>>>(dst, dpitch,
        value, width, height));
  }
}

template<arithmetic T, arithmetic U>
void memset(T* dst, const int dpitch, const U* value, const int width,
    const int height) {
  if (width > 0 && height > 0) {
    auto grid = make_grid(width, height);
    auto block = make_block(width, height);
    CUDA_LAUNCH(kernel_memset<<<grid,block,0,stream>>>(dst, dpitch,
        value, width, height));
  }
}

}
