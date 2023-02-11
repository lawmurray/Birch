/**
 * @file
 * 
 * cuBLAS boilerplate.
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"

#include <cublas_v2.h>
#include <cassert>

/*
 * Call a cuBLAS function and assert success.
 */
#define CUBLAS_CHECK(call) \
    { \
      cublasStatus_t err = call; \
      if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << cublasGetStatusString(err) << std::endl; \
      } \
      assert(err == CUBLAS_STATUS_SUCCESS); \
      if (CUDA_SYNC) { \
        cudaError_t err = cudaStreamSynchronize(stream); \
        if (err != cudaSuccess) { \
          std::cerr << cudaGetErrorString(err) << std::endl; \
        } \
        assert(err == cudaSuccess); \
      } \
    }

namespace numbirch {
/**
 * @internal
 * 
 * cuBLAS handle for each host thread.
 */
extern thread_local cublasHandle_t cublasHandle;

/**
 * @internal
 * 
 * cuBLAS worksapce for each host thread.
 */
extern thread_local void* cublasWorkspace;

/*
 * cuBLAS function access for single and double precision.
 */
template<class T>
struct cublas {
  //
};
template<>
struct cublas<double> {
  static constexpr auto copy = cublasDcopy;
  static constexpr auto dot = cublasDdot;
  static constexpr auto gemv = cublasDgemv;
  static constexpr auto gemm = cublasDgemm;
  static constexpr auto trmv = cublasDtrmv;
  static constexpr auto trmm = cublasDtrmm;
  static constexpr auto trsv = cublasDtrsv;
  static constexpr auto trsm = cublasDtrsm;
};
template<>
struct cublas<float> {
  static constexpr auto copy = cublasScopy;
  static constexpr auto dot = cublasSdot;
  static constexpr auto gemv = cublasSgemv;
  static constexpr auto gemm = cublasSgemm;
  static constexpr auto trmv = cublasStrmv;
  static constexpr auto trmm = cublasStrmm;
  static constexpr auto trsv = cublasStrsv;
  static constexpr auto trsm = cublasStrsm;
};

/*
 * Scalars.
 */
extern double* oneD;
extern double* zeroD;
extern float* oneS;
extern float* zeroS;
extern int* oneI;
extern int* zeroI;

/*
 * Scalar access for single and double precision.
 */
template<class T>
struct scalar {
  //
};
template<>
struct scalar<double> {
  static constexpr double*& one = oneD;
  static constexpr double*& zero = zeroD;
};
template<>
struct scalar<float> {
  static constexpr float*& one = oneS;
  static constexpr float*& zero = zeroS;
};
template<>
struct scalar<int> {
  static constexpr int*& one = oneI;
  static constexpr int*& zero = zeroI;
};

/**
 * @internal
 * 
 * Initialize cuBLAS integrations. This should be called during init() by the
 * backend.
 */
void cublas_init();

/**
 * @internal
 * 
 * Terminate cuBLAS integrations. This should be called during term() by the
 * backend.
 */
void cublas_term();

}
