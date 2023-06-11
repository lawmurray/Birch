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

#define UNARY_GRAD(f, R) \
    UNARY_GRAD_FIRST(f, R, real) \
    UNARY_GRAD_FIRST(f, R, int) \
    UNARY_GRAD_FIRST(f, R, bool)
#define UNARY_GRAD_FIRST(f, R, T) \
    UNARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 2)) \
    UNARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 1)) \
    UNARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 0)) \
    UNARY_GRAD_SIG(f, R, T)
#define UNARY_GRAD_SIG(f, R, T) \
    template real_t<T> f<T>(const real_t<T>&, const R<T>&, const T&);

#define UNARY_ARITHMETIC_GRAD(f) UNARY_GRAD(f, implicit_t)
#define UNARY_REAL_GRAD(f) UNARY_GRAD(f, real_t)
#define UNARY_BOOL_GRAD(f) UNARY_GRAD(f, bool_t)

namespace numbirch {
UNARY_ARITHMETIC_GRAD(neg_grad)
UNARY_ARITHMETIC_GRAD(abs_grad)
UNARY_REAL_GRAD(acos_grad)
UNARY_REAL_GRAD(asin_grad)
UNARY_REAL_GRAD(atan_grad)
UNARY_ARITHMETIC_GRAD(ceil_grad)
UNARY_REAL_GRAD(cos_grad)
UNARY_REAL_GRAD(cosh_grad)
UNARY_REAL_GRAD(exp_grad)
UNARY_REAL_GRAD(expm1_grad)
UNARY_ARITHMETIC_GRAD(floor_grad)
UNARY_BOOL_GRAD(isfinite_grad)
UNARY_BOOL_GRAD(isinf_grad)
UNARY_BOOL_GRAD(isnan_grad)
UNARY_REAL_GRAD(lfact_grad)
UNARY_REAL_GRAD(lgamma_grad)
UNARY_REAL_GRAD(log_grad)
UNARY_REAL_GRAD(log1p_grad)
UNARY_BOOL_GRAD(logical_not_grad)
UNARY_ARITHMETIC_GRAD(rectify_grad)
UNARY_ARITHMETIC_GRAD(round_grad)
UNARY_REAL_GRAD(sin_grad)
UNARY_REAL_GRAD(sinh_grad)
UNARY_REAL_GRAD(sqrt_grad)
UNARY_REAL_GRAD(tan_grad)
UNARY_REAL_GRAD(tanh_grad)
}
