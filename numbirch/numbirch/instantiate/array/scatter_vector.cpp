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

#define SCATTER_VECTOR(f) \
    SCATTER_VECTOR_SIG(f, real) \
    SCATTER_VECTOR_SIG(f, int) \
    SCATTER_VECTOR_SIG(f, bool)
#define SCATTER_VECTOR_SIG(f, T) \
    template Array<T,1> f<T>(const Array<T,1>& x, const Array<int,1>& y, \
        const int n); \
    template Array<real,1> f##_grad1(const Array<real,1>& g, \
        const Array<T,1>& x, const Array<int,1>& y, const int n); \
    template Array<real,1> f##_grad2(const Array<real,1>& g, \
        const Array<T,1>& x, const Array<int,1>& y, const int n);

namespace numbirch {
SCATTER_VECTOR(scatter)
}
