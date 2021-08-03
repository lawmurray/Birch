/**
 * @file
 */
#include "numbirch/cuda/cublas.hpp"

namespace numbirch {

thread_local cublasHandle_t cublasHandle;
thread_local void* cublasWorkspace;

void cublas_init() {
  #pragma omp parallel
  {
    size_t size = 1 << 21; // 4 MB as recommended in CUBLAS docs
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
    CUDA_CHECK(cudaMalloc(&cublasWorkspace, size));
    CUBLAS_CHECK(cublasSetWorkspace(cublasHandle, cublasWorkspace, size));
    CUBLAS_CHECK(cublasSetPointerMode(cublasHandle,
        CUBLAS_POINTER_MODE_DEVICE));
    CUBLAS_CHECK(cublasSetAtomicsMode(cublasHandle,
        CUBLAS_ATOMICS_ALLOWED));
  }

}

void cublas_term() {
  #pragma omp parallel
  {
    CUBLAS_CHECK(cublasDestroy(cublasHandle));
    CUDA_CHECK(cudaFree(cublasWorkspace));
  }
}

}
