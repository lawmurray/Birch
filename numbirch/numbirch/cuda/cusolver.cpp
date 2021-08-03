/**
 * @file
 */
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"

namespace numbirch {

thread_local cusolverDnHandle_t cusolverDnHandle;
thread_local cusolverDnParams_t cusolverDnParams;

double* oneD;
double* zeroD;
float* oneS;
float* zeroS;

void cusolver_init() {
  /* double-precision scalars for cuSOLVER */
  double oneD1 = 1.0;
  double zeroD1 = 0.0;
  CUDA_CHECK(cudaMalloc(&oneD, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&zeroD, sizeof(double)));  
  CUBLAS_CHECK(cublasSetVector(1, sizeof(double), &oneD1, 1, oneD, 1));
  CUBLAS_CHECK(cublasSetVector(1, sizeof(double), &zeroD1, 1, zeroD, 1));

  /* single-precision scalars for cuSOLVER */
  float oneS1 = 1.0;
  float zeroS1 = 0.0;
  CUDA_CHECK(cudaMalloc(&oneS, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&zeroS, sizeof(float)));  
  CUBLAS_CHECK(cublasSetVector(1, sizeof(float), &oneS1, 1, oneS, 1));
  CUBLAS_CHECK(cublasSetVector(1, sizeof(float), &zeroS1, 1, zeroS, 1));

  #pragma omp parallel
  {
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
  }
  CUDA_CHECK(cudaFree(zeroD));
  CUDA_CHECK(cudaFree(oneD));
  CUDA_CHECK(cudaFree(zeroS));
  CUDA_CHECK(cudaFree(oneS));
}

}
