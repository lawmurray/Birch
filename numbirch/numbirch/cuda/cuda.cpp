/**
 * @file
 */
#include "numbirch/cuda/cuda.hpp"

namespace numbirch {

thread_local int device = 0;
thread_local cudaStream_t stream = cudaStreamPerThread;

void cuda_init() {
  #pragma omp parallel
  {
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
}

void cuda_term() {
  #pragma omp parallel
  {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
}

}
