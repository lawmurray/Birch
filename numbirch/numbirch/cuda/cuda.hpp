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
static const int CUDA_PREFERRED_BLOCK_SIZE = 512;

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
  dim3 block;
  block.x = std::min(n, CUDA_PREFERRED_BLOCK_SIZE);
  block.y = 1;
  block.z = 1;
  return block;
}

/*
 * Configure thread block size for a matrix transformation.
 */
inline dim3 make_block(const int m, const int n) {
  dim3 block;
  block.x = std::min(m, CUDA_PREFERRED_BLOCK_SIZE);
  block.y = std::min(n, CUDA_PREFERRED_BLOCK_SIZE/(int)block.x);
  block.z = 1;
  return block;
}

/*
 * Configure grid size for a vector transformation.
 */
inline dim3 make_grid(const int n) {
  dim3 block = make_block(n);
  dim3 grid;
  grid.x = (n + block.x - 1)/block.x;
  grid.y = 1;
  grid.z = 1;
  return grid;
}

/*
 * Configure grid size for a matrix transformation.
 */
inline dim3 make_grid(const int m, const int n) {
  dim3 block = make_block(m, n);
  dim3 grid;
  grid.x = (n + block.x - 1)/block.x;
  grid.y = (m + block.y - 1)/block.y;
  grid.z = 1;
  return grid;
}

}
