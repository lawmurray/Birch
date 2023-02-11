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

/**
 * @internal
 * 
 * @def CUDA_SYNC
 * 
 * If true, all CUDA calls are synchronous, which can be helpful to determine
 * precisely which call causes an error. Can also set the environment variable
 * CUDA_LAUNCH_BLOCKING=1.
 */
#define CUDA_SYNC 0

/**
 * @internal
 * 
 * @def CUDA_CHECK
 * 
 * Call a cuda* function and assert success.
 */
#define CUDA_CHECK(call) \
    { \
      cudaError_t err = call; \
      if (err != cudaSuccess) { \
        std::cerr << cudaGetErrorString(err) << std::endl; \
      } \
      assert(err == cudaSuccess); \
      if (CUDA_SYNC) { \
        cudaError_t err = cudaStreamSynchronize(stream); \
        if (err != cudaSuccess) { \
          std::cerr << cudaGetErrorString(err) << std::endl; \
        } \
        assert(err == cudaSuccess); \
      } \
    }

/**
 * @internal
 * 
 * @def CUDA_LAUNCH
 * 
 * Launch a CUDA kernel and assert success.
 */
#define CUDA_LAUNCH(call...) \
    { \
      call; \
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
 * Event used by each host thread.
 */
extern thread_local cudaEvent_t event;

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
extern thread_local unsigned max_blocks;

/**
 * @internal
 * 
 * Maximum (and preferred) thread block size for a kernel launch
 * configuration. The setting of 256 is appropriate for recent generations of
 * GPUs, as well as being the maximum number of threads that cuRAND supports
 * with a single MRG32k3a generator in a single block.
 */
static const unsigned MAX_BLOCK_SIZE = 256;

/**
 * @internal
 * 
 * Configure thread block size for a transformation.
 */
inline dim3 make_block(const unsigned m, const unsigned n) {
  dim3 block{0, 0, 0};
  block.x = std::min(m, MAX_BLOCK_SIZE);
  if (block.x > 0) {
    block.y = std::min(n, MAX_BLOCK_SIZE/block.x);
    block.z = 1;
  }
  return block;
}

/**
 * @internal
 * 
 * Configure thread grid size for a transformation.
 */
inline dim3 make_grid(const unsigned m, const unsigned n) {
  dim3 block = make_block(m, n);
  dim3 grid{0, 0, 0};
  if (block.x > 0 && block.y > 0) {
    grid.x = std::min((m + block.x - 1)/block.x, max_blocks);
    grid.y = std::min((n + block.y - 1)/block.y, max_blocks/grid.x);
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
