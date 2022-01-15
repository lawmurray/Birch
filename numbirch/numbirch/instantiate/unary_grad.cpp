/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/unary.hpp"

#define UNARY_GRAD(f) \
    UNARY_GRAD_FIRST(f, real) \
    UNARY_GRAD_FIRST(f, int) \
    UNARY_GRAD_FIRST(f, bool)
#define UNARY_GRAD_FIRST(f, R) \
    UNARY_GRAD_SECOND(f, R, real) \
    UNARY_GRAD_SECOND(f, R, int) \
    UNARY_GRAD_SECOND(f, R, bool)
#define UNARY_GRAD_SECOND(f, R, T) \
    UNARY_GRAD_SIG(f, ARRAY(R, 2), ARRAY(T, 2)) \
    UNARY_GRAD_SIG(f, ARRAY(R, 1), ARRAY(T, 1)) \
    UNARY_GRAD_SIG(f, ARRAY(R, 0), ARRAY(T, 0)) \
    UNARY_GRAD_SIG(f, ARRAY(R, 0), T) \
    UNARY_GRAD_SIG(f, R, ARRAY(T, 0)) \
    UNARY_GRAD_SIG(f, R, T)
#define UNARY_GRAD_SIG(f, R, T) \
    template default_t<T> f<R,T,int>(const default_t<T>&, \
        const explicit_t<R,T>&, const T&);

#define UNARY_GRAD_FLOATING_POINT(f) \
    UNARY_GRAD_FIRST(f, real)

namespace numbirch {
UNARY_GRAD(not_grad)

UNARY_GRAD(abs_grad)
UNARY_GRAD_FLOATING_POINT(acos_grad)
UNARY_GRAD_FLOATING_POINT(asin_grad)
UNARY_GRAD_FLOATING_POINT(atan_grad)
UNARY_GRAD(ceil_grad)
UNARY_GRAD_FLOATING_POINT(cos_grad)
UNARY_GRAD_FLOATING_POINT(cosh_grad)
UNARY_GRAD_FLOATING_POINT(exp_grad)
UNARY_GRAD_FLOATING_POINT(expm1_grad)
UNARY_GRAD(floor_grad)
UNARY_GRAD_FLOATING_POINT(lfact_grad)
UNARY_GRAD_FLOATING_POINT(lgamma_grad)
UNARY_GRAD_FLOATING_POINT(log_grad)
UNARY_GRAD_FLOATING_POINT(log1p_grad)
UNARY_GRAD(rectify_grad)
UNARY_GRAD(round_grad)
UNARY_GRAD_FLOATING_POINT(sin_grad)
UNARY_GRAD_FLOATING_POINT(sinh_grad)
UNARY_GRAD_FLOATING_POINT(sqrt_grad)
UNARY_GRAD_FLOATING_POINT(tan_grad)
UNARY_GRAD_FLOATING_POINT(tanh_grad)

}
