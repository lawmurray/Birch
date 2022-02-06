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
#define BINARY_SCALAR_DIVIDE_FIRST(f, T) \
    BINARY_SCALAR_DIVIDE_SECOND(f, T, real) \
    BINARY_SCALAR_DIVIDE_SECOND(f, T, int) \
    BINARY_SCALAR_DIVIDE_SECOND(f, T, bool)
#define BINARY_SCALAR_DIVIDE_SECOND(f, T, U) \
    BINARY_SCALAR_DIVIDE_SIG(f, ARRAY(T, 2), ARRAY(U, 0)) \
    BINARY_SCALAR_DIVIDE_SIG(f, ARRAY(T, 1), ARRAY(U, 0)) \
    BINARY_SCALAR_DIVIDE_SIG(f, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_SCALAR_DIVIDE_SIG(f, ARRAY(T, 2), U) \
    BINARY_SCALAR_DIVIDE_SIG(f, ARRAY(T, 1), U) \
    BINARY_SCALAR_DIVIDE_SIG(f, ARRAY(T, 0), U) \
    BINARY_SCALAR_DIVIDE_SIG(f, T, ARRAY(U, 0))
#define BINARY_SCALAR_DIVIDE_SIG(f, T, U) \
    template implicit_t<T,U> f<T,U,int>(const T&, const U&);

namespace numbirch {
BINARY_SCALAR_DIVIDE(operator/)
}
