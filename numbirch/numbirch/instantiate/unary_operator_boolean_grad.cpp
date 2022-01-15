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

#define UNARY_OPERATOR_BOOLEAN_GRAD(f) \
    UNARY_OPERATOR_BOOLEAN_GRAD_FIRST(f, real) \
    UNARY_OPERATOR_BOOLEAN_GRAD_FIRST(f, int) \
    UNARY_OPERATOR_BOOLEAN_GRAD_FIRST(f, bool)
#define UNARY_OPERATOR_BOOLEAN_GRAD_FIRST(f, T) \
    UNARY_OPERATOR_BOOLEAN_GRAD_SIG(f, ARRAY(T, 2)) \
    UNARY_OPERATOR_BOOLEAN_GRAD_SIG(f, ARRAY(T, 1)) \
    UNARY_OPERATOR_BOOLEAN_GRAD_SIG(f, ARRAY(T, 0))
#define UNARY_OPERATOR_BOOLEAN_GRAD_SIG(f, T) \
    template default_t<T> f<T,int>(const default_t<T>&, \
        const explicit_t<bool,T>&, const T&);

namespace numbirch {
UNARY_OPERATOR_BOOLEAN_GRAD(not_grad)
}
