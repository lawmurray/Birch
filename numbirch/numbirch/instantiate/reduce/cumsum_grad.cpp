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

#define REDUCE_CUMSUM_GRAD(f) \
    REDUCE_CUMSUM_GRAD_FIRST(f, real) \
    REDUCE_CUMSUM_GRAD_FIRST(f, int) \
    REDUCE_CUMSUM_GRAD_FIRST(f, bool)
#define REDUCE_CUMSUM_GRAD_FIRST(f, T) \
    REDUCE_CUMSUM_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    REDUCE_CUMSUM_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    REDUCE_CUMSUM_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    REDUCE_CUMSUM_GRAD_SIG(f, T)
#define REDUCE_CUMSUM_GRAD_SIG(f, T) \
    template real_t<T> f(const Array<real,0>&, const T&, const T&);

namespace numbirch {
REDUCE_CUMSUM_GRAD(cumsum_grad)
}
