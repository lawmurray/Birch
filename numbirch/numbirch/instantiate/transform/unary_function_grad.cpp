/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/transform.hpp"

#define UNARY_FUNCTION_GRAD(f) \
    UNARY_FUNCTION_GRAD_FIRST(f, real) \
    UNARY_FUNCTION_GRAD_FIRST(f, int) \
    UNARY_FUNCTION_GRAD_FIRST(f, bool)
#define UNARY_FUNCTION_GRAD_FIRST(f, T) \
    UNARY_FUNCTION_GRAD_SIG(f, ARRAY(T, 2)) \
    UNARY_FUNCTION_GRAD_SIG(f, ARRAY(T, 1)) \
    UNARY_FUNCTION_GRAD_SIG(f, ARRAY(T, 0)) \
    UNARY_FUNCTION_GRAD_SIG(f, T)
#define UNARY_FUNCTION_GRAD_SIG(f, T) \
    template default_t<T> f<T,int>(const default_t<T>&, const T&, const T&);

namespace numbirch {
UNARY_FUNCTION_GRAD(abs_grad)
UNARY_FUNCTION_GRAD(ceil_grad)
UNARY_FUNCTION_GRAD(floor_grad)
UNARY_FUNCTION_GRAD(rectify_grad)
UNARY_FUNCTION_GRAD(round_grad)
}
