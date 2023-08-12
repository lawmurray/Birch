/**
 * @file
 */
#include "numbirch/common/reduce.inl"
#ifdef BACKEND_CUDA
#include "numbirch/cuda/reduce.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/reduce.inl"
#endif

#define CUMSUM_GRAD(f) \
    CUMSUM_GRAD_FIRST(f, real) \
    CUMSUM_GRAD_FIRST(f, int) \
    CUMSUM_GRAD_FIRST(f, bool)
#define CUMSUM_GRAD_FIRST(f, T) \
    CUMSUM_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    CUMSUM_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    CUMSUM_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    CUMSUM_GRAD_SIG(f, T)
#define CUMSUM_GRAD_SIG(f, T) \
    template real_t<T> f(const real_t<T>&, const T&);

namespace numbirch {
CUMSUM_GRAD(cumsum_grad)
}
