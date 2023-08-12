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

#define COUNT_GRAD(f) \
    COUNT_GRAD_FIRST(f, real) \
    COUNT_GRAD_FIRST(f, int) \
    COUNT_GRAD_FIRST(f, bool)
#define COUNT_GRAD_FIRST(f, T) \
    COUNT_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    COUNT_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    COUNT_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    COUNT_GRAD_SIG(f, T)
#define COUNT_GRAD_SIG(f, T) \
    template real_t<T> f(const Array<real,0>&, const T&);

namespace numbirch {
COUNT_GRAD(count_grad)
}
