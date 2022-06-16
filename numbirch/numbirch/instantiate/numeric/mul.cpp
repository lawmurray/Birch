/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.inl"
#endif

#define BINARY_MATRIX(f) \
    BINARY_MATRIX_SIG(f, real)
#define BINARY_MATRIX_SIG(f, T) \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&); \
    template Array<T,1> f(const Array<T,2>&, const Array<T,1>&);

namespace numbirch {
BINARY_MATRIX(operator*)
BINARY_MATRIX(trimul)
}
