/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#endif
#include "numbirch/common/array.inl"

#define GATHER_MATRIX(f) \
    GATHER_MATRIX_SIG(f, real) \
    GATHER_MATRIX_SIG(f, int) \
    GATHER_MATRIX_SIG(f, bool)
#define GATHER_MATRIX_SIG(f, T) \
    template Array<T,2> f<T,int>(const Array<T,2>& A, const Array<int,2>& I, \
        const Array<int,2>& J);

namespace numbirch {
GATHER_MATRIX(gather)
}
