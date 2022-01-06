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

/**
 * @internal
 * 
 * @def UNARY_GRAD
 * 
 * Explicitly instantiate the gradient of a unary transformation `f`.
 */
#define UNARY_GRAD(f) \
    UNARY_GRAD_FIRST(f, real)
#define UNARY_GRAD_FIRST(f, G) \
    UNARY_GRAD_SECOND(f, G, real) \
    UNARY_GRAD_SECOND(f, G, int) \
    UNARY_GRAD_SECOND(f, G, bool)
#define UNARY_GRAD_SECOND(f, G, T) \
    UNARY_GRAD_SIG(f, ARRAY(G, 2), ARRAY(T, 2)) \
    UNARY_GRAD_SIG(f, ARRAY(G, 1), ARRAY(T, 1)) \
    UNARY_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 0)) \
    UNARY_GRAD_SIG(f, ARRAY(G, 0), T) \
    UNARY_GRAD_SIG(f, G, ARRAY(T, 0)) \
    UNARY_GRAD_SIG(f, G, T)
#define UNARY_GRAD_SIG(f, G, T) \
    template default_t<G,T> f<G,T,int>(const G&, const T&);

namespace numbirch {
UNARY_GRAD(not_grad)

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
UNARY_GRAD(lfact_grad)
UNARY_GRAD(lgamma_grad)
UNARY_GRAD(log_grad)
UNARY_GRAD(log1p_grad)
UNARY_GRAD(rectify_grad)
UNARY_GRAD(round_grad)
UNARY_GRAD(sin_grad)
UNARY_GRAD(sinh_grad)
UNARY_GRAD(sqrt_grad)
UNARY_GRAD(tan_grad)
UNARY_GRAD(tanh_grad)

}
