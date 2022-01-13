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
 * Explicitly instantiate a unary reduction function `f`.
 */
#define REDUCE_MATRIX(f) \
    REDUCE_MATRIX_SIG(f, real)
#define REDUCE_MATRIX_SIG(f, T) \
    template Array<T,0> f(const Array<T,2>&);

/**
 * @internal
 * 
 * @def REDUCE_MATRIX_GRAD
 * 
 * Explicitly instantiate the gradient of a unary reduction function `f`.
 */
#define REDUCE_MATRIX_GRAD(f) \
    REDUCE_MATRIX_GRAD_SIG(f, real)
#define REDUCE_MATRIX_GRAD_SIG(f, T) \
    template Array<T,2> f(const Array<T,0>& g, const Array<T,2>&);

namespace numbirch {
REDUCE_MATRIX(lcholdet)
REDUCE_MATRIX_GRAD(lcholdet_grad)
REDUCE_MATRIX(ldet)
}
