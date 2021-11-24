/**
 * @file
 */
#include "numbirch/cuda/cuda.hpp"

namespace numbirch {

thread_local int device = 0;
thread_local cudaStream_t stream = 0;

void cuda_init() {
  #pragma omp parallel
  {
    int value = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    /* check device support */
    CUDA_CHECK(cudaDeviceGetAttribute(&value,
        cudaDevAttrConcurrentKernels, device));
    assert(value && "device with concurrent kernel support required");
    CUDA_CHECK(cudaDeviceGetAttribute(&value,
        cudaDevAttrConcurrentManagedAccess, device));
    assert(value && "device with concurrent managed memory support required");

    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
}

void cuda_term() {
  #pragma omp parallel
  {
    /* don't use CUDA_CHECK here, because it tries to use stream after
     * destruction when CUDA_SYNC is true */
    cudaError_t err = cudaStreamDestroy(stream);
    assert(err == cudaSuccess);
  }
}

}
