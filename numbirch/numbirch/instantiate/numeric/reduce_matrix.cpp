/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.inl"
#endif

#define REDUCE_MATRIX(f) \
    REDUCE_MATRIX_SIG(f, real)
#define REDUCE_MATRIX_SIG(f, T) \
    template Array<T,0> f(const Array<T,2>&);

namespace numbirch {
REDUCE_MATRIX(lcholdet)
REDUCE_MATRIX(ldet)
}
