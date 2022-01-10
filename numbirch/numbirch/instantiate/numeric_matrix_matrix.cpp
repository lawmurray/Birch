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
 * @def MATRIX_MATRIX
 * 
 * Explicitly instantiate a binary function `f` over floating point matrices.
 */
#define MATRIX_MATRIX(f) \
    MATRIX_MATRIX_SIG(f, real)
#define MATRIX_MATRIX_SIG(f, T) \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&);

namespace numbirch {
MATRIX_MATRIX(operator*)
MATRIX_MATRIX(cholmul)
MATRIX_MATRIX(cholouter)
MATRIX_MATRIX(cholsolve)
MATRIX_MATRIX(solve)
}
