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
 * @def CUDA_PREFERRED_BLOCK_SIZE
 * 
 * Preferred thread block size for CUDA kernels.
 */
#define CUDA_PREFERRED_BLOCK_SIZE 512

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

template<class T>
struct vector_element_functor {
  vector_element_functor(T* x, int incx) :
      x(x),
      incx(incx) {
    //
  }
  __host__ __device__
  T operator()(const int i) const {
    return x[i*incx];
  }
  T* x;
  int incx;
};

template<class T>
struct matrix_element_functor {
  matrix_element_functor(T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  __host__ __device__
  T operator()(const int i) const {
    int c = i/m;
    int r = i - c*m;
    return A[r + c*ldA];
  }
  T* A;
  int m;
  int ldA;
};

template<class T>
struct matrix_transpose_element_functor {
  matrix_transpose_element_functor(T* A, int m, int ldA) :
      A(A),
      m(m),
      ldA(ldA) {
    //
  }
  __host__ __device__
  T operator()(const int i) const {
    int r = i/m;
    int c = i - r*m;
    return A[r + c*ldA];
  }
  T* A;
  int m;
  int ldA;
};

template<class T = double>
struct negate_functor {
  __host__ __device__
  T operator()(const T x) const {
    return -x;
  }
};

template<class T = double>
struct plus_functor {
  __host__ __device__
  T operator()(const T x, const T y) const {
    return x + y;
  }
};

template<class T = double>
struct minus_functor {
  __host__ __device__
  T operator()(const T x, const T y) const {
    return x - y;
  }
};

template<class T = double>
struct multiplies_functor {
  __host__ __device__
  T operator()(const T x, const T y) const {
    return x*y;
  }
};

template<class T = double>
struct divides_functor {
  __host__ __device__
  T operator()(const T x, const T y) const {
    return x/y;
  }
};

template<class T = double>
struct scalar_multiplies_functor {
  scalar_multiplies_functor(T a) :
      a(a) {
    //
  }
  __host__ __device__
  T operator()(const T x) const {
    return x*a;
  }
  T a;
};

template<class T = double>
struct scalar_divides_functor {
  scalar_divides_functor(T a) :
      a(a) {
    //
  }
  __host__ __device__
  T operator()(const T x) const {
    return x/a;
  }
  T a;
};

template<class T = double>
struct log_abs_functor {
  __host__ __device__
  T operator()(const T x) const {
    return std::log(std::abs(x));
  }
};

template<class T = double>
struct log_functor {
  __host__ __device__
  T operator()(const T x) const {
    return std::log(x);
  }
};

/**
 * Configure thread block size for a vector transformation.
 */
inline dim3 make_block(const int n) {
  dim3 block;
  block.x = std::min(n, CUDA_PREFERRED_BLOCK_SIZE);
  block.y = 1;
  block.z = 1;
  return block;
}

/**
 * Configure thread block size for a matrix transformation.
 */
inline dim3 make_block(const int m, const int n) {
  dim3 block;
  block.x = std::min(m, CUDA_PREFERRED_BLOCK_SIZE);
  block.y = CUDA_PREFERRED_BLOCK_SIZE/block.x;
  block.z = 1;
  return block;
}

/**
 * Configure grid size for a vector transformation.
 */
inline dim3 make_grid(const int n) {
  dim3 block = make_block(n);
  dim3 grid;
  grid.x = (n + block.x - 1)/block.x;
  grid.y = 1;
  grid.z = 1;
  return grid;
}

/**
 * Configure grid size for a matrix transformation.
 */
inline dim3 make_grid(const int m, const int n) {
  dim3 block = make_block(m, n);
  dim3 grid;
  grid.x = (n + block.x - 1)/block.x;
  grid.y = (m + block.y - 1)/block.y;
  grid.z = 1;
  return grid;
}
