/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.hpp"
#endif

#define MATRIX_VECTOR(f) \
    MATRIX_VECTOR_SIG(f, real)
#define MATRIX_VECTOR_SIG(f, T) \
    template Array<T,1> f(const Array<T,2>&, const Array<T,1>&);

namespace numbirch {
MATRIX_VECTOR(operator*)
MATRIX_VECTOR(trimul)
MATRIX_VECTOR(cholsolve)
MATRIX_VECTOR(solve)
}
