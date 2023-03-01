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

#define SCATTER_MATRIX(f) \
    SCATTER_MATRIX_SIG(f, real) \
    SCATTER_MATRIX_SIG(f, int) \
    SCATTER_MATRIX_SIG(f, bool)
#define SCATTER_MATRIX_SIG(f, T) \
    template Array<T,2> f<T,int>(const Array<T,2>& A, const Array<int,2>& I, \
        const Array<int,2>& J, const int m, const int n);

namespace numbirch {
SCATTER_MATRIX(scatter)
}
