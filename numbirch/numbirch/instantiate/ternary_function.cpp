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
#include "numbirch/common/ternary.hpp"

/**
 * @internal
 * 
 * @def TERNARY_ARITHMETIC
 * 
 * Explicitly instantiate a ternary transformation `f` where the return type
 * is any arithmetic type.
 */
#define TERNARY_ARITHMETIC(f) \
    TERNARY_FIRST(f, double) \
    TERNARY_FIRST(f, float) \
    TERNARY_FIRST(f, int) \
    TERNARY_FIRST(f, bool)
#define TERNARY_FIRST(f, R) \
    TERNARY_SECOND(f, R, double) \
    TERNARY_SECOND(f, R, float) \
    TERNARY_SECOND(f, R, int) \
    TERNARY_SECOND(f, R, bool)
#define TERNARY_SECOND(f, R, T) \
    TERNARY_THIRD(f, R, T, double) \
    TERNARY_THIRD(f, R, T, float) \
    TERNARY_THIRD(f, R, T, int) \
    TERNARY_THIRD(f, R, T, bool)
#define TERNARY_THIRD(f, R, T, U) \
    TERNARY_DIM(f, R, T, U, double) \
    TERNARY_DIM(f, R, T, U, float) \
    TERNARY_DIM(f, R, T, U, int) \
    TERNARY_DIM(f, R, T, U, bool)
#define TERNARY_DIM(f, R, T, U, V) \
    TERNARY_SIG(f, R, ARRAY(T, 2), ARRAY(U, 2), ARRAY(V, 2)) \
    TERNARY_SIG(f, R, ARRAY(T, 1), ARRAY(U, 1), ARRAY(V, 1)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0), ARRAY(V, 0)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0), V) \
    TERNARY_SIG(f, R, ARRAY(T, 0), U, ARRAY(V, 0)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), U, V) \
    TERNARY_SIG(f, R, T, ARRAY(U, 0), ARRAY(V, 0)) \
    TERNARY_SIG(f, R, T, ARRAY(U, 0), V) \
    TERNARY_SIG(f, R, T, U, ARRAY(V, 0)) \
    TERNARY_SIG(f, R, T, U, V)
#define TERNARY_SIG(f, R, T, U, V) \
    template explicit_t<R,implicit_t<T,U,V>> f<R,T,U,V,int>(const T&, const U&, const V&);

/**
 * @internal
 * 
 * @def TERNARY_FLOATING_POINT
 * 
 * Explicitly instantiate a ternary transformation `f` where the return type
 * is any floating point type.
 */
#define TERNARY_FLOATING_POINT(f) \
    TERNARY_FIRST(f, double) \
    TERNARY_FIRST(f, float)

namespace numbirch {
TERNARY_FLOATING_POINT(ibeta)
}
