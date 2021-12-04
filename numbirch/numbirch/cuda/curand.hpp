/**
 * @file
 * 
 * cuRAND boilerplate.
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"

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
 * 32-bit pseudorandom number generator for each host thread.
 */
extern thread_local std::mt19937 rng32;

/**
 * @internal
 * 
 * 64-bit pseudorandom number generator for each host thread.
 */
extern thread_local std::mt19937_64 rng64;

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

/*
 * Templated access to required functions and objects for single and double
 * precision.
 */
template<class T>
struct curand {
  //
};
template<>
struct curand<double> {
  static auto& rng() {
    return rng64;
  }
  static constexpr auto uniform = curand_uniform_double;
  static constexpr auto normal = curand_normal_double;
  static constexpr auto poisson = curand_poisson_double;
};
template<>
struct curand<float> {
  static auto& rng() {
    return rng32;
  }
  static constexpr auto uniform = curand_uniform;
  static constexpr auto normal = curand_normal;
  static constexpr auto poisson = curand_poisson;
};

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
