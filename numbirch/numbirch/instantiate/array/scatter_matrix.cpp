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
    template Array<T,2> f<T>(const Array<T,2>& A, const Array<int,2>& I, \
        const Array<int,2>& J, const int m, const int n); \
    template Array<real,2> f##_grad1(const Array<real,2>& G, \
        const Array<T,2>& C, const Array<T,2>& A, const Array<int,2>& I, \
        const Array<int,2>& J, const int m, const int n); \
    template Array<real,2> f##_grad2(const Array<real,2>& G, \
        const Array<T,2>& C, const Array<T,2>& A, const Array<int,2>& I, \
        const Array<int,2>& J, const int m, const int n); \
    template Array<real,2> f##_grad3(const Array<real,2>& G, \
        const Array<T,2>& C, const Array<T,2>& A, const Array<int,2>& I, \
        const Array<int,2>& J, const int m, const int n);

namespace numbirch {
SCATTER_MATRIX(scatter)
}
