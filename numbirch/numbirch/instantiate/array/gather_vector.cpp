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

#define GATHER_VECTOR(f) \
    GATHER_VECTOR_SIG(f, real) \
    GATHER_VECTOR_SIG(f, int) \
    GATHER_VECTOR_SIG(f, bool)
#define GATHER_VECTOR_SIG(f, T) \
    template Array<T,1> f<T,int>(const Array<T,1>& x, const Array<int,1>& y);

namespace numbirch {
GATHER_VECTOR(gather)
}
