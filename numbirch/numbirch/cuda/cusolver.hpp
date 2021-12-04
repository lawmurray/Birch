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
 * Call a cuSOLVER function and assert success.
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

/*
 * Call a cuSOLVER function and assert success, on both return code and info
 * code (the latter only if CUDA_SYNC is enabled).
 */
#define CUSOLVER_CHECK_INFO(call) \
    { \
      cusolverStatus_t err = call; \
      assert(err == CUSOLVER_STATUS_SUCCESS); \
      if (CUDA_SYNC) { \
        cudaError_t err = cudaStreamSynchronize(stream); \
        assert(err == cudaSuccess); \
        assert(*info == 0); \
      } \
    }

namespace numbirch {

extern thread_local cusolverDnHandle_t cusolverDnHandle;
extern thread_local cusolverDnParams_t cusolverDnParams;
extern thread_local int* info;

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

/**
 * @internal
 * 
 * Initialize cuSOLVER integrations. This should be called during init() by
 * the backend.
 */
void cusolver_init();

/**
 * @internal
 * 
 * Terminate cuSOLVER integrations. This should be called during term() by the
 * backend.
 */
void cusolver_term();
}