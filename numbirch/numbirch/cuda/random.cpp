/**
 * @file
 */
#include "numbirch/cuda/curand.hpp"

#if HAVE_OMP_H
#include <omp.h>
#endif

namespace numbirch {

__global__ void kernel_seed(const int s, curandState_t* rngs) {
  auto k = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(s, k, 0, &rngs[k]);
}

void seed(const int s) {
  #pragma omp parallel num_threads(omp_get_max_threads())
  {
    /* seed device generators */
    #if HAVE_OMP_H
    int n = omp_get_thread_num();
    int N = omp_get_max_threads();
    #else
    int n = 0;
    int N = 1;
    #endif
    dim3 grid{1, 1, 1}, block{1, 1, 1};
    block.x = MAX_BLOCK_SIZE;
    grid.x = max_blocks;
    CUDA_LAUNCH(kernel_seed<<<grid,block,0,stream>>>(s*N + n, rngs));

    /* seed host generator; fine to use the same seed here, and seed as the
     * device generators above, as these are all different algorithms and/or
     * parameterizations of them */
    rng32.seed(s*N + n);
    rng64.seed(s*N + n);
  }
}

void seed() {
  std::random_device rd;
  seed(rd());
}

Array<real,1> convolve(const Array<real,1>& p, const Array<real,1>& q) {

}

}
