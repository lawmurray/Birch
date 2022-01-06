/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.hpp"
#endif

/**
 * @internal
 * 
 * @def REDUCE_MATRIX
 * 
 * Explicitly instantiate a unary reduction function `f` for matrices of
 * floating point types only.
 */
#define REDUCE_MATRIX(f) \
    REDUCE_MATRIX_SIG(f, real)
#define REDUCE_MATRIX_SIG(f, T) \
    template Array<T,0> f(const Array<T,2>&);

namespace numbirch {
REDUCE_MATRIX(lcholdet)
REDUCE_MATRIX(ldet)
}
