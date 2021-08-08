/**
 * @file
 */
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"

namespace numbirch {

thread_local cusolverDnHandle_t cusolverDnHandle;
thread_local cusolverDnParams_t cusolverDnParams;
thread_local int* info = nullptr;

void cusolver_init() {
  #pragma omp parallel
  {
    CUDA_CHECK(cudaMallocHost(&info, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverDnHandle));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverDnHandle, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&cusolverDnParams));
  }
}

void cusolver_term() {
  #pragma omp parallel
  {
    CUSOLVER_CHECK(cusolverDnDestroyParams(cusolverDnParams));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolverDnHandle));
    CUDA_CHECK(cudaFreeHost(info));
  }
}

}
