/**
 * @file
 */
#pragma once

#include "numbirch/random.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/curand.hpp"
#include "numbirch/macro.hpp"

namespace numbirch {

/**
 * @internal
 * 
 * Get the pseudorandom number generator for the current device thread.
 */
__device__ inline auto get_rng() {
  auto x = threadIdx.x + blockIdx.x*blockDim.x;
  auto y = threadIdx.y + blockIdx.y*blockDim.y;
  return rngs[x + y*gridDim.x*blockDim.x];
}

template<class R>
struct simulate_gaussian_functor {
  template<class T, class U>
  NUMBIRCH_HOST R operator()(const T μ, const U σ2) const {
    return std::normal_distribution<R>(μ, std::sqrt(σ2))(stl<R>::rng());
  }
  template<class T, class U>
  NUMBIRCH_DEVICE R operator()(const T μ, const U σ2) {
    return μ + std::sqrt(σ2)*curand<R>::normal(get_rng());
  }
};

}
