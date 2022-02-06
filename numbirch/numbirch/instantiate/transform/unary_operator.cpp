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

#define UNARY_OPERATOR(f) \
    UNARY_OPERATOR_FIRST(f, real) \
    UNARY_OPERATOR_FIRST(f, int) \
    UNARY_OPERATOR_FIRST(f, bool)
#define UNARY_OPERATOR_FIRST(f, T) \
    UNARY_OPERATOR_SIG(f, ARRAY(T, 2)) \
    UNARY_OPERATOR_SIG(f, ARRAY(T, 1)) \
    UNARY_OPERATOR_SIG(f, ARRAY(T, 0))
#define UNARY_OPERATOR_SIG(f, T) \
    template T f<T,int>(const T&);

namespace numbirch {
UNARY_OPERATOR(operator-)
}
