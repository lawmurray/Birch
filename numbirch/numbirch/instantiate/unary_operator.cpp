/**
 * @file
 */
#include "numbirch/numeric.hpp"

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
 * @def UNARY_ARITHMETIC_OPERATOR
 * 
 * Explicitly instantiate a unary transformation `f` where the return type is
 * any arithmetic type. This version is used for operators, where the overload
 * for basic types is omitted, as this is not allowed in C++ (e.g.
 * `operator-(double)`).
 */
#define UNARY_OPERATOR(f) \
    UNARY_OPERATOR_FIRST(f, double) \
    UNARY_OPERATOR_FIRST(f, float) \
    UNARY_OPERATOR_FIRST(f, int) \
    UNARY_OPERATOR_FIRST(f, bool)
#define UNARY_OPERATOR_RETURN(f) \
    UNARY_OPERATOR_FIRST(f, double) \
    UNARY_OPERATOR_FIRST(f, float)
#define UNARY_OPERATOR_FIRST(f, R) \
    UNARY_OPERATOR_DIM(f, R, double) \
    UNARY_OPERATOR_DIM(f, R, float) \
    UNARY_OPERATOR_DIM(f, R, int) \
    UNARY_OPERATOR_DIM(f, R, bool)
#define UNARY_OPERATOR_DIM(f, R, T) \
    UNARY_OPERATOR_SIG(f, R, ARRAY(T, 2)) \
    UNARY_OPERATOR_SIG(f, R, ARRAY(T, 1)) \
    UNARY_OPERATOR_SIG(f, R, ARRAY(T, 0))
#define UNARY_OPERATOR_SIG(f, R, T) \
    template explicit_t<R,T> f<R,T,int>(const T&);

namespace numbirch {
UNARY_OPERATOR(operator+)
UNARY_OPERATOR(operator-)
UNARY_OPERATOR(operator!)
}
