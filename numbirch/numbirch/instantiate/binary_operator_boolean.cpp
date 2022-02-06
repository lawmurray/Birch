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

#define BINARY_OPERATOR_BOOLEAN(f) \
    BINARY_OPERATOR_BOOLEAN_FIRST(f, real) \
    BINARY_OPERATOR_BOOLEAN_FIRST(f, int) \
    BINARY_OPERATOR_BOOLEAN_FIRST(f, bool)
#define BINARY_OPERATOR_BOOLEAN_FIRST(f, T) \
    BINARY_OPERATOR_BOOLEAN_SECOND(f, T, real) \
    BINARY_OPERATOR_BOOLEAN_SECOND(f, T, int) \
    BINARY_OPERATOR_BOOLEAN_SECOND(f, T, bool)
#define BINARY_OPERATOR_BOOLEAN_SECOND(f, T, U) \
    BINARY_OPERATOR_BOOLEAN_SIG(f, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_OPERATOR_BOOLEAN_SIG(f, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_OPERATOR_BOOLEAN_SIG(f, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_OPERATOR_BOOLEAN_SIG(f, ARRAY(T, 0), U) \
    BINARY_OPERATOR_BOOLEAN_SIG(f, T, ARRAY(U, 0))
#define BINARY_OPERATOR_BOOLEAN_SIG(f, T, U) \
    template explicit_t<bool,T,U> f<T,U,int>(const T&, const U&);

namespace numbirch {
BINARY_OPERATOR_BOOLEAN(operator&&)
BINARY_OPERATOR_BOOLEAN(operator||)
BINARY_OPERATOR_BOOLEAN(operator==)
BINARY_OPERATOR_BOOLEAN(operator!=)
BINARY_OPERATOR_BOOLEAN(operator<)
BINARY_OPERATOR_BOOLEAN(operator<=)
BINARY_OPERATOR_BOOLEAN(operator>)
BINARY_OPERATOR_BOOLEAN(operator>=)
}
