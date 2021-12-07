/**
 * @file
 * 
 * CUDA boilerplate.
 */
#pragma once

#include <cassert>
#include <iostream>

/* for recent versions of CUDA, disables warnings about diag_suppress being
 * deprecated in favor of nv_diag_suppress */
#pragma nv_diag_suppress 20236

/* for recent versions of CUDA, disables warnings about diag_suppress being
 * deprecated in favor of nv_diag_suppress */
#pragma nv_diag_suppress 20012

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
        if (err != cudaSuccess) { \
          std::cerr << cudaGetErrorString(err) << std::endl; \
        } \
        assert(err == cudaSuccess); \
      } \
    }

namespace numbirch {
/**
 * @internal
 * 
 * Device used by each host thread.
 */
extern thread_local int device;

/**
 * @internal
 * 
 * Stream used by each host thread.
 */
extern thread_local cudaStream_t stream;

/**
 * @internal
 * 
 * Maximum (and preferred) number of blocks for a kernel launch configuration.
 * Threads are able to perform multiple units of work, so this does not limit
 * the task size, but a maximum is required in order to initialize a
 * sufficient number of pseudorandom number generator states.
 * 
 * Currently this is set to four times the multiprocessor count on
 * the device used by each host thread, capped at 200, as cuRAND includes this
 * many parameterizations for the MRG32k3a generator.
 */
extern thread_local int max_blocks;

/**
 * @internal
 * 
 * Maximum (and preferred) thread block size for a kernel launch
 * configuration. The setting of 256 is appropriate for recent generations of
 * GPUs, as well as being the maximum number of threads that cuRAND supports
 * with a single MRG32k3a generator in a single block.
 */
static const int MAX_BLOCK_SIZE = 256;

/**
 * @internal
 * 
 * Configure thread block size for a vector transformation.
 */
inline dim3 make_block(const int n) {
  dim3 block{0, 0, 0};
  block.x = std::min(n, MAX_BLOCK_SIZE);
  if (block.x > 0) {
    block.y = 1;
    block.z = 1;
  }
  return block;
}

/**
 * @internal
 * 
 * Configure thread block size for a matrix transformation.
 */
inline dim3 make_block(const int m, const int n) {
  dim3 block{0, 0, 0};
  block.x = std::min(m, MAX_BLOCK_SIZE);
  if (block.x > 0) {
    block.y = std::min(n, MAX_BLOCK_SIZE/(int)block.x);
    block.z = 1;
  }
  return block;
}

/**
 * @internal
 * 
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

/**
 * @internal
 * 
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

/**
 * @internal
 * 
 * Initialize CUDA integrations. This should be called during init() by the
 * backend.
 */
void cuda_init();

/**
 * @internal
 * 
 * Terminate CUDA integrations. This should be called during term() by the
 * backend.
 */
void cuda_term();

}
