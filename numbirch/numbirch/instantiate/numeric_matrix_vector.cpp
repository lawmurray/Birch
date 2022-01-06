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
 * @def MATRIX_VECTOR
 * 
 * Explicitly instantiate a binary function `f` over a floating point matrix
 * and vector. Use cases include solve(), matrix-vector multiplication.
 */
#define MATRIX_VECTOR(f) \
    MATRIX_VECTOR_SIG(f, real)
#define MATRIX_VECTOR_SIG(f, T) \
    template Array<T,1> f(const Array<T,2>&, const Array<T,1>&);

namespace numbirch {
MATRIX_VECTOR(operator*)
MATRIX_VECTOR(cholmul)
MATRIX_VECTOR(cholsolve)
MATRIX_VECTOR(inner)
MATRIX_VECTOR(solve)
}
