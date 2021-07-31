/**
 * @file
 * 
 * Some common macros for CUDA.
 */
#pragma once

/**
 * @def CUDA_SYNC
 * 
 * If true, all CUDA calls are synchronous, which can be helpful to determine
 * precisely which call causes an error.
 */
#define CUDA_SYNC 0

/**
 * @def CUDA_CHECK
 * 
 * Call a cuda* function and assert success.
 */
#define CUDA_CHECK(call) \
    { \
      cudaError_t err = call; \
      assert(err == cudaSuccess); \
      if (CUDA_SYNC) { \
        cudaError_t err = cudaStreamSynchronize(stream); \
        assert(err == cudaSuccess); \
      } \
    }

/**
 * @def CUBLAS_CHECK
 * 
 * Call a cublas* function and assert success.
 */
#define CUBLAS_CHECK(call) \
    { \
      cublasStatus_t err = call; \
      assert(err == CUBLAS_STATUS_SUCCESS); \
      if (CUDA_SYNC) { \
        cudaError_t err = cudaStreamSynchronize(stream); \
        assert(err == cudaSuccess); \
      } \
    }

/**
 * @def CUSOLVER_CHECK
 * 
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
