/**
 * @file
 * 
 * cuRAND boilerplate.
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"
#include "numbirch/common/stl.hpp"
#include "numbirch/macro.hpp"

#include <curand_kernel.h>
#include <cassert>

/*
 * Call a cuRAND function and assert success.
 */
#define CURAND_CHECK(call) \
    { \
      curandStatus_t err = call; \
      assert(err == CURAND_STATUS_SUCCESS); \
      if (CUDA_SYNC) { \
        cudaError_t err = cudaStreamSynchronize(stream); \
        assert(err == cudaSuccess); \
      } \
    }


namespace numbirch {
/**
 * @internal
 * 
 * Device pseudorandom number generators for each host thread. The XORWOW
 * generator of cuRAND is used. The MTGP32 generator may be preferred for its
 * longer period (XORWOW: 2^190, MTGP32: 2^11214), but is more difficult to
 * use when implementing rejection sampling algorithms (e.g. custom
 * implementation for gamma variates here), as it can only generate variates
 * for all threads simultaneously. Furthermore, on inspection of
 * curand_uniform.h, it is documented as generating double-precision variates
 * with only 32 random bits, which does not seem ideal.
 */
extern thread_local curandState_t* rngs;

/**
 * @internal
 * 
 * Get the pseudorandom number generator for the current device thread.
 */
NUMBIRCH_DEVICE inline curandState_t* curand_rng() {
  extern __shared__ curandState_t* shared[];
  curandState_t* rngs = shared[0];
  auto x = threadIdx.x + blockIdx.x*blockDim.x;
  auto y = threadIdx.y + blockIdx.y*blockDim.y;
  return &rngs[x + y*gridDim.x*blockDim.x];
}

/**
 * @internal
 * 
 * Initialize cuRAND integrations. This should be called during init() by the
 * backend.
 */
void curand_init();

/**
 * @internal
 * 
 * Terminate cuRAND integrations. This should be called during term() by the
 * backend.
 */
void curand_term();

}
