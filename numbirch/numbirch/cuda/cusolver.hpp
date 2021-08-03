/**
 * @file
 * 
 * cuSOLVER boilerplate.
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"

#include <cusolverDn.h>
#include <cassert>

/*
 * Call a cusolver* function and assert success.
 */
#define CUSOLVER_CHECK(call) \
    { \
      cusolverStatus_t err = call; \
      assert(err == CUSOLVER_STATUS_SUCCESS); \
      if (CUDA_SYNC) { \
        cudaError_t err = cudaStreamSynchronize(stream); \
        assert(err == cudaSuccess); \
      } \
    }

namespace numbirch {

extern thread_local cusolverDnHandle_t cusolverDnHandle;
extern thread_local cusolverDnParams_t cusolverDnParams;

/*
 * Scalars.
 */
extern double* oneD;
extern double* zeroD;
extern float* oneS;
extern float* zeroS;

/*
 * Info flag.
 */
static int* info = nullptr;

/*
 * cuSOLVER scalar access for single and double precision.
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

/*
 * cuSOLVER requirements for single and double precision.
 */
template<class T>
struct cusolver {
  //
};
template<>
struct cusolver<double> {
  static constexpr auto CUDA_R = CUDA_R_64F;
};
template<>
struct cusolver<float> {
  static constexpr auto CUDA_R = CUDA_R_32F;
};

/*
 * Initialize cuSOLVER integrations. This should be called during init() by
 * the backend.
 */
void cusolver_init();

/*
 * Terminate cuSOLVER integrations. This should be called during term() by the
 * backend.
 */
void cusolver_term();
}