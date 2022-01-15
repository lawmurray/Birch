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

#define UNARY_FUNCTION_REAL_GRAD(f) \
    UNARY_FUNCTION_REAL_GRAD_FIRST(f, real) \
    UNARY_FUNCTION_REAL_GRAD_FIRST(f, int) \
    UNARY_FUNCTION_REAL_GRAD_FIRST(f, bool)
#define UNARY_FUNCTION_REAL_GRAD_FIRST(f, T) \
    UNARY_FUNCTION_REAL_GRAD_SIG(f, ARRAY(T, 2)) \
    UNARY_FUNCTION_REAL_GRAD_SIG(f, ARRAY(T, 1)) \
    UNARY_FUNCTION_REAL_GRAD_SIG(f, ARRAY(T, 0)) \
    UNARY_FUNCTION_REAL_GRAD_SIG(f, T)
#define UNARY_FUNCTION_REAL_GRAD_SIG(f, T) \
    template default_t<T> f<T,int>(const default_t<T>&, const default_t<T>&, \
        const T&);

namespace numbirch {
UNARY_FUNCTION_REAL_GRAD(acos_grad)
UNARY_FUNCTION_REAL_GRAD(asin_grad)
UNARY_FUNCTION_REAL_GRAD(atan_grad)
UNARY_FUNCTION_REAL_GRAD(cos_grad)
UNARY_FUNCTION_REAL_GRAD(cosh_grad)
UNARY_FUNCTION_REAL_GRAD(lfact_grad)
UNARY_FUNCTION_REAL_GRAD(lgamma_grad)
UNARY_FUNCTION_REAL_GRAD(log_grad)
UNARY_FUNCTION_REAL_GRAD(log1p_grad)
UNARY_FUNCTION_REAL_GRAD(sin_grad)
UNARY_FUNCTION_REAL_GRAD(sinh_grad)
UNARY_FUNCTION_REAL_GRAD(sqrt_grad)
UNARY_FUNCTION_REAL_GRAD(tan_grad)
UNARY_FUNCTION_REAL_GRAD(tanh_grad)
}
