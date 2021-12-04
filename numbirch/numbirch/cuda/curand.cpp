/**
 * @file
 */
#include "numbirch/cuda/cuda.hpp"

namespace numbirch {
thread_local std::mt19937 rng32;
thread_local std::mt19937_64 rng64;
thread_local curandState_t* rngs = nullptr;

void curand_init() {
  #pragma omp parallel
  {
    CUDA_CHECK(cudaMalloc(&rngs, max_blocks*MAX_BLOCK_SIZE*
        sizeof(curandState_t)));
  }
}

void curand_term() {
  #pragma omp parallel
  {
    CUDA_CHECK(cudaFree(rngs));
  }
}

}
