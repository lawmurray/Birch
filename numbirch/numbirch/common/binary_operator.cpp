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
#include "numbirch/common/binary.hpp"

/**
 * @internal
 * 
 * @def BINARY_OPERATOR
 * 
 * Explicitly instantiate a binary operator `f` where the return type is
 * any arithmetic type. This version is used for operators, where the overload
 * for basic types is omitted, as this is not allowed in C++ (e.g.
 * `operator+(double, double)`).
 */
#define BINARY_OPERATOR(f) \
    BINARY_OPERATOR_FIRST(f, double) \
    BINARY_OPERATOR_FIRST(f, float) \
    BINARY_OPERATOR_FIRST(f, int) \
    BINARY_OPERATOR_FIRST(f, bool)
#define BINARY_OPERATOR_FIRST(f, R) \
    BINARY_OPERATOR_SECOND(f, R, double) \
    BINARY_OPERATOR_SECOND(f, R, float) \
    BINARY_OPERATOR_SECOND(f, R, int) \
    BINARY_OPERATOR_SECOND(f, R, bool)
#define BINARY_OPERATOR_SECOND(f, R, T) \
    BINARY_OPERATOR_DIM(f, R, T, double) \
    BINARY_OPERATOR_DIM(f, R, T, float) \
    BINARY_OPERATOR_DIM(f, R, T, int) \
    BINARY_OPERATOR_DIM(f, R, T, bool)
#define BINARY_OPERATOR_DIM(f, R, T, U) \
    BINARY_OPERATOR_SIG(f, R, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_OPERATOR_SIG(f, R, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_OPERATOR_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_OPERATOR_SIG(f, R, ARRAY(T, 0), U) \
    BINARY_OPERATOR_SIG(f, R, T, ARRAY(U, 0))
#define BINARY_OPERATOR_SIG(f, R, T, U) \
    template explicit_t<R,implicit_t<T,U>> f<R,T,U,int>(const T&, const U&);

namespace numbirch {
BINARY_OPERATOR(operator+)
BINARY_OPERATOR(operator-)
BINARY_OPERATOR(operator&&)
BINARY_OPERATOR(operator||)
BINARY_OPERATOR(operator==)
BINARY_OPERATOR(operator!=)
BINARY_OPERATOR(operator<)
BINARY_OPERATOR(operator<=)
BINARY_OPERATOR(operator>)
BINARY_OPERATOR(operator>=)
}
