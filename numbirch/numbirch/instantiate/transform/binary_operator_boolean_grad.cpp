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

#define BINARY_OPERATOR_BOOLEAN_GRAD(f) \
    BINARY_OPERATOR_BOOLEAN_GRAD_FIRST(f, real) \
    BINARY_OPERATOR_BOOLEAN_GRAD_FIRST(f, int) \
    BINARY_OPERATOR_BOOLEAN_GRAD_FIRST(f, bool)
#define BINARY_OPERATOR_BOOLEAN_GRAD_FIRST(f, T) \
    BINARY_OPERATOR_BOOLEAN_GRAD_SECOND(f, T, real) \
    BINARY_OPERATOR_BOOLEAN_GRAD_SECOND(f, T, int) \
    BINARY_OPERATOR_BOOLEAN_GRAD_SECOND(f, T, bool)
#define BINARY_OPERATOR_BOOLEAN_GRAD_SECOND(f, T, U) \
    BINARY_OPERATOR_BOOLEAN_GRAD_SIG(f, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_OPERATOR_BOOLEAN_GRAD_SIG(f, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_OPERATOR_BOOLEAN_GRAD_SIG(f, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_OPERATOR_BOOLEAN_GRAD_SIG(f, ARRAY(T, 0), U) \
    BINARY_OPERATOR_BOOLEAN_GRAD_SIG(f, T, ARRAY(U, 0))
#define BINARY_OPERATOR_BOOLEAN_GRAD_SIG(f, T, U) \
    template std::pair<default_t<T>,default_t<U>> f<T,U,int>( \
        const default_t<T,U>&, const explicit_t<bool,T,U>&, const T&, \
        const U&);

namespace numbirch {
BINARY_OPERATOR_BOOLEAN_GRAD(and_grad)
BINARY_OPERATOR_BOOLEAN_GRAD(or_grad)
BINARY_OPERATOR_BOOLEAN_GRAD(equal_grad)
BINARY_OPERATOR_BOOLEAN_GRAD(not_equal_grad)
BINARY_OPERATOR_BOOLEAN_GRAD(less_grad)
BINARY_OPERATOR_BOOLEAN_GRAD(less_or_equal_grad)
BINARY_OPERATOR_BOOLEAN_GRAD(greater_grad)
BINARY_OPERATOR_BOOLEAN_GRAD(greater_or_equal_grad)
}
