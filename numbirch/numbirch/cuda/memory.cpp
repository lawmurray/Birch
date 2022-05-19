/**
 * @file
 */
#include "numbirch/memory.hpp"
#include "numbirch/random.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/curand.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"

namespace numbirch {

void init() {
  cuda_init();
  cublas_init();
  cusolver_init();
  curand_init();
  jemalloc_init();
  seed();
}

void wait() {
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

void term() {
  jemalloc_term();
  curand_term();
  cusolver_term();
  cublas_term();
  cuda_term();
}

void memcpy(void* dst, const size_t dpitch, const void* src,
    const size_t spitch, const size_t width, const size_t height) {
  CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
      cudaMemcpyDefault, stream));
}

}
