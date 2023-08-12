/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#endif
#include "numbirch/common/reduce.inl"

#define REDUCE_GRAD(f) \
    REDUCE_GRAD_FIRST(f, real) \
    REDUCE_GRAD_FIRST(f, int) \
    REDUCE_GRAD_FIRST(f, bool)
#define REDUCE_GRAD_FIRST(f, T) \
    REDUCE_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    REDUCE_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    REDUCE_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    REDUCE_GRAD_SIG(f, T)
#define REDUCE_GRAD_SIG(f, T) \
    template real_t<T> f(const Array<real,0>&, const Array<value_t<T>,0>&, \
        const T&);

namespace numbirch {
REDUCE_GRAD(min_grad)
REDUCE_GRAD(max_grad)
}
