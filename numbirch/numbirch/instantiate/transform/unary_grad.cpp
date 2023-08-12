/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#endif
#include "numbirch/common/transform.inl"

#define UNARY_GRAD(f) \
    UNARY_GRAD_FIRST(f, real) \
    UNARY_GRAD_FIRST(f, int) \
    UNARY_GRAD_FIRST(f, bool)
#define UNARY_GRAD_FIRST(f, T) \
    UNARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    UNARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    UNARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    UNARY_GRAD_SIG(f, T)
#define UNARY_GRAD_SIG(f, T) \
    template real_t<T> f<T>(const real_t<T>&, const T&);

namespace numbirch {
UNARY_GRAD(neg_grad)
UNARY_GRAD(abs_grad)
UNARY_GRAD(acos_grad)
UNARY_GRAD(asin_grad)
UNARY_GRAD(atan_grad)
UNARY_GRAD(ceil_grad)
UNARY_GRAD(cos_grad)
UNARY_GRAD(cosh_grad)
UNARY_GRAD(exp_grad)
UNARY_GRAD(expm1_grad)
UNARY_GRAD(floor_grad)
UNARY_GRAD(isfinite_grad)
UNARY_GRAD(isinf_grad)
UNARY_GRAD(isnan_grad)
UNARY_GRAD(lfact_grad)
UNARY_GRAD(lgamma_grad)
UNARY_GRAD(log_grad)
UNARY_GRAD(log1p_grad)
UNARY_GRAD(logical_not_grad)
UNARY_GRAD(rectify_grad)
UNARY_GRAD(round_grad)
UNARY_GRAD(sin_grad)
UNARY_GRAD(sinh_grad)
UNARY_GRAD(sqrt_grad)
UNARY_GRAD(tan_grad)
UNARY_GRAD(tanh_grad)
}
