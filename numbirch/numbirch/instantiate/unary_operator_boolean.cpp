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

#define UNARY_OPERATOR_BOOLEAN(f) \
    UNARY_OPERATOR_BOOLEAN_FIRST(f, real) \
    UNARY_OPERATOR_BOOLEAN_FIRST(f, int) \
    UNARY_OPERATOR_BOOLEAN_FIRST(f, bool)
#define UNARY_OPERATOR_BOOLEAN_FIRST(f, T) \
    UNARY_OPERATOR_BOOLEAN_SIG(f, ARRAY(T, 2)) \
    UNARY_OPERATOR_BOOLEAN_SIG(f, ARRAY(T, 1)) \
    UNARY_OPERATOR_BOOLEAN_SIG(f, ARRAY(T, 0))
#define UNARY_OPERATOR_BOOLEAN_SIG(f, T) \
    template explicit_t<bool,T> f<T,int>(const T&);

namespace numbirch {
UNARY_OPERATOR_BOOLEAN(operator!)
}
