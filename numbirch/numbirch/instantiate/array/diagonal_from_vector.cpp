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

#define DIAGONAL(f) \
    DIAGONAL_SIG(f, real) \
    DIAGONAL_SIG(f, int) \
    DIAGONAL_SIG(f, bool)
#define DIAGONAL_SIG(f, T) \
    template Array<T,2> f(const Array<T,1>& x); \
    template Array<real,1> f##_grad(const Array<real,2>& g, \
        const Array<T,1>& x);

namespace numbirch {
DIAGONAL(diagonal)
}
