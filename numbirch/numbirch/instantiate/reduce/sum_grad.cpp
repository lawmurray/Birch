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

#define SUM_GRAD(f) \
    SUM_GRAD_FIRST(f, real) \
    SUM_GRAD_FIRST(f, int) \
    SUM_GRAD_FIRST(f, bool)
#define SUM_GRAD_FIRST(f, T) \
    SUM_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    SUM_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    SUM_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    SUM_GRAD_SIG(f, T)
#define SUM_GRAD_SIG(f, T) \
    template real_t<T> f(const Array<real,0>&, const T&);

namespace numbirch {
SUM_GRAD(sum_grad)
}
