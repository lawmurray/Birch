/**
 * @file
 */
#include "numbirch/cuda/curand.hpp"

#if HAVE_OMP_H
#include <omp.h>
#endif

namespace numbirch {
thread_local curandState_t* rngs = nullptr;

void curand_init() {
  #pragma omp parallel
  {
    CUDA_CHECK(cudaMalloc(&rngs, max_blocks*MAX_BLOCK_SIZE*
        sizeof(curandState_t)));
  }
}

void curand_term() {
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    CUDA_CHECK(cudaFree(rngs));
  }
}

}
