/**
 * @file
 */
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
 * @def BINARY_SCALAR_DIVIDE
 * 
 * Explicitly instantiate a binary scalar division. As this is used for
 * operators, the overload for basic types is omitted, as this is not allowed
 * in C++ (e.g. `operator+(double, double)`).
 */
#define BINARY_SCALAR_DIVIDE(f) \
    BINARY_SCALAR_DIVIDE_FIRST(f, real) \
    BINARY_SCALAR_DIVIDE_FIRST(f, int) \
    BINARY_SCALAR_DIVIDE_FIRST(f, bool)
#define BINARY_SCALAR_DIVIDE_FIRST(f, R) \
    BINARY_SCALAR_DIVIDE_SECOND(f, R, real) \
    BINARY_SCALAR_DIVIDE_SECOND(f, R, int) \
    BINARY_SCALAR_DIVIDE_SECOND(f, R, bool)
#define BINARY_SCALAR_DIVIDE_SECOND(f, R, T) \
    BINARY_SCALAR_DIVIDE_THIRD(f, R, T, real) \
    BINARY_SCALAR_DIVIDE_THIRD(f, R, T, int) \
    BINARY_SCALAR_DIVIDE_THIRD(f, R, T, bool)
#define BINARY_SCALAR_DIVIDE_THIRD(f, R, T, U) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 2), ARRAY(U, 0)) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 1), ARRAY(U, 0)) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 2), U) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 1), U) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 0), U) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, T, ARRAY(U, 0))
#define BINARY_SCALAR_DIVIDE_SIG(f, R, T, U) \
    template explicit_t<R,T,U> f<R,T,U,int>(const T&, const U&);

namespace numbirch {
BINARY_SCALAR_DIVIDE(operator/)
}
