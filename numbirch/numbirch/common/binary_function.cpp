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
 * @def BINARY_ARITHMETIC
 * 
 * Explicitly instantiate a binary transformation `f` where the return type is
 * any arithmetic type.
 */
#define BINARY_ARITHMETIC(f) \
    BINARY_FIRST(f, double) \
    BINARY_FIRST(f, float) \
    BINARY_FIRST(f, int) \
    BINARY_FIRST(f, bool)
#define BINARY_FIRST(f, R) \
    BINARY_SECOND(f, R, double) \
    BINARY_SECOND(f, R, float) \
    BINARY_SECOND(f, R, int) \
    BINARY_SECOND(f, R, bool)
#define BINARY_SECOND(f, R, T) \
    BINARY_DIM(f, R, T, double) \
    BINARY_DIM(f, R, T, float) \
    BINARY_DIM(f, R, T, int) \
    BINARY_DIM(f, R, T, bool)
#define BINARY_DIM(f, R, T, U) \
    BINARY_SIG(f, R, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_SIG(f, R, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_SIG(f, R, ARRAY(T, 0), U) \
    BINARY_SIG(f, R, T, ARRAY(U, 0)) \
    BINARY_SIG(f, R, T, U)
#define BINARY_SIG(f, R, T, U) \
    template explicit_t<R,implicit_t<T,U>> f<R,T,U,int>(const T&, const U&);

/**
 * @internal
 * 
 * @def BINARY_FLOATING_POINT
 * 
 * Explicitly instantiate a binary transformation `f` where the return type is
 * any floating point type.
 */
#define BINARY_FLOATING_POINT(f) \
    BINARY_FIRST(f, double) \
    BINARY_FIRST(f, float)

namespace numbirch {
BINARY_ARITHMETIC(copysign)
BINARY_FLOATING_POINT(digamma)
BINARY_FLOATING_POINT(gamma_p)
BINARY_FLOATING_POINT(gamma_q)
BINARY_ARITHMETIC(hadamard)
BINARY_FLOATING_POINT(lbeta)
BINARY_FLOATING_POINT(lchoose)
BINARY_FLOATING_POINT(lgamma)
BINARY_FLOATING_POINT(pow)
}
