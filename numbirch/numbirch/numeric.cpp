/**
 * @file
 */
#include "numbirch/numeric.hpp"

#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif

/**
 * @internal
 * 
 * @def ARRAY
 * 
 * Constructs the type `Array<T,D>`.
 */
#define ARRAY(T, D) Array<T,D>

/**
 * @internal
 * 
 * @def UNARY_TRANSFORM_INSTANTIATION
 * 
 * Explicitly instantiate a binary transformation `f` for parameter type `T`.
 */
#define UNARY_TRANSFORM_INSTANTIATION(f, T) \
    template T f(const T&);

/**
 * @internal
 * 
 * @def UNARY_TRANSFORM_INSTANTIATIONS
 * 
 * Explicitly instantiate a unary transformation `f` for scalar parameter
 * type `T` and all array sizes.
 */
#define UNARY_TRANSFORM_INSTANTIATIONS(f, T) \
    UNARY_TRANSFORM_INSTANTIATION(f, ARRAY(T, 2)) \
    UNARY_TRANSFORM_INSTANTIATION(f, ARRAY(T, 1)) \
    UNARY_TRANSFORM_INSTANTIATION(f, ARRAY(T, 0))

/**
 * @internal
 * 
 * @def UNARY
 * 
 * Explicit instantiate a binary transformation `f` for all types.
 */
#define UNARY_TRANSFORM(f) \
    UNARY_TRANSFORM_INSTANTIATIONS(f, double) \
    UNARY_TRANSFORM_INSTANTIATIONS(f, float) \
    UNARY_TRANSFORM_INSTANTIATIONS(f, int) \
    UNARY_TRANSFORM_INSTANTIATIONS(f, bool)

/**
 * @internal
 * 
 * @def BINARY_TRANSFORM_INSTANTIATION
 * 
 * Explicitly instantiate a binary transformation `f` for parameter types `T`
 * and `U`.
 */
#define BINARY_TRANSFORM_INSTANTIATION(f, T, U) \
    template promote_t<T,U> f(const T&, const U&);

/**
 * @internal
 * 
 * @def BINARY_TRANSFORM_INSTANTIATIONS
 * 
 * Explicitly instantiate a binary transformation `f` for scalar parameter
 * types `T` and `U`, for all array sizes.
 */
#define BINARY_TRANSFORM_INSTANTIATIONS(f, T, U) \
    BINARY_TRANSFORM_INSTANTIATION(f, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_TRANSFORM_INSTANTIATION(f, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_TRANSFORM_INSTANTIATION(f, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_TRANSFORM_INSTANTIATION(f, ARRAY(T, 0), U) \
    BINARY_TRANSFORM_INSTANTIATION(f, T, ARRAY(U, 0))

/**
 * @internal
 * 
 * @def BINARY_TRANSFORM
 * 
 * Explicitly instantiate a binary transformation `f` for all pairs of
 * compatible types.
 */
#define BINARY_TRANSFORM(f) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, double, double) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, double, float) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, double, int) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, double, bool) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, float, double) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, float, float) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, float, int) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, float, bool) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, int, double) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, int, float) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, int, int) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, int, bool) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, bool, double) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, bool, float) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, bool, int) \
    BINARY_TRANSFORM_INSTANTIATIONS(f, bool, bool)

/**
 * @internal
 * 
 * @def BINARY_COMPARE_INSTANTIATION
 * 
 * Explicitly instantiate a binary comparison `f` for parameter types `T`
 * and `U`.
 */
#define BINARY_COMPARE_INSTANTIATION(f, T, U) \
    template Array<bool,dimension_v<T>> f(const T&, const U&);

/**
 * @internal
 * 
 * @def BINARY_COMPARE_INSTANTIATIONS
 * 
 * Explicitly instantiate a binary comparison `f` for scalar parameter
 * types `T` and `U`, for all array sizes.
 */
#define BINARY_COMPARE_INSTANTIATIONS(f, T, U) \
    BINARY_COMPARE_INSTANTIATION(f, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_COMPARE_INSTANTIATION(f, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_COMPARE_INSTANTIATION(f, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_COMPARE_INSTANTIATION(f, ARRAY(T, 0), U) \
    BINARY_COMPARE_INSTANTIATION(f, T, ARRAY(U, 0))

/**
 * @internal
 * 
 * @def BINARY_COMPARE
 * 
 * Explicitly instantiate a binary comparison `f` for all pairs of
 * compatible types.
 */
#define BINARY_COMPARE(f) \
    BINARY_COMPARE_INSTANTIATIONS(f, double, double) \
    BINARY_COMPARE_INSTANTIATIONS(f, double, float) \
    BINARY_COMPARE_INSTANTIATIONS(f, double, int) \
    BINARY_COMPARE_INSTANTIATIONS(f, double, bool) \
    BINARY_COMPARE_INSTANTIATIONS(f, float, double) \
    BINARY_COMPARE_INSTANTIATIONS(f, float, float) \
    BINARY_COMPARE_INSTANTIATIONS(f, float, int) \
    BINARY_COMPARE_INSTANTIATIONS(f, float, bool) \
    BINARY_COMPARE_INSTANTIATIONS(f, int, double) \
    BINARY_COMPARE_INSTANTIATIONS(f, int, float) \
    BINARY_COMPARE_INSTANTIATIONS(f, int, int) \
    BINARY_COMPARE_INSTANTIATIONS(f, int, bool) \
    BINARY_COMPARE_INSTANTIATIONS(f, bool, double) \
    BINARY_COMPARE_INSTANTIATIONS(f, bool, float) \
    BINARY_COMPARE_INSTANTIATIONS(f, bool, int) \
    BINARY_COMPARE_INSTANTIATIONS(f, bool, bool)

/**
 * @internal
 * 
 * @def BINARY_SCALAR_INSTANTIATION
 * 
 * Explicitly instantiate a binary comparison `f` for parameter types `T`
 * and `U`.
 */
#define BINARY_SCALAR_INSTANTIATION(f, T, U) \
    template promote_t<T,U> f(const T&, const U&);

/**
 * @internal
 * 
 * @def BINARY_SCALAR_INSTANTIATIONS
 * 
 * Explicitly instantiate a binary transformation `f` requiring a scalar
 * second argument, for scalar parameter types types `T` and `U`, for all
 * array sizes.
 */
#define BINARY_SCALAR_INSTANTIATIONS(f, T, U) \
    BINARY_SCALAR_INSTANTIATION(f, ARRAY(T, 2), ARRAY(U, 0)) \
    BINARY_SCALAR_INSTANTIATION(f, ARRAY(T, 2), U) \
    BINARY_SCALAR_INSTANTIATION(f, ARRAY(T, 1), ARRAY(U, 0)) \
    BINARY_SCALAR_INSTANTIATION(f, ARRAY(T, 1), U) \
    BINARY_SCALAR_INSTANTIATION(f, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_SCALAR_INSTANTIATION(f, ARRAY(T, 0), U) \
    BINARY_SCALAR_INSTANTIATION(f, T, ARRAY(U, 0))

/**
 * @internal
 * 
 * @def BINARY_SCALAR
 * 
 * Explicitly instantiate a binary comparison `f` for all pairs of
 * compatible types.
 */
#define BINARY_SCALAR(f) \
    BINARY_SCALAR_INSTANTIATIONS(f, double, double) \
    BINARY_SCALAR_INSTANTIATIONS(f, double, float) \
    BINARY_SCALAR_INSTANTIATIONS(f, double, int) \
    BINARY_SCALAR_INSTANTIATIONS(f, double, bool) \
    BINARY_SCALAR_INSTANTIATIONS(f, float, double) \
    BINARY_SCALAR_INSTANTIATIONS(f, float, float) \
    BINARY_SCALAR_INSTANTIATIONS(f, float, int) \
    BINARY_SCALAR_INSTANTIATIONS(f, float, bool) \
    BINARY_SCALAR_INSTANTIATIONS(f, int, double) \
    BINARY_SCALAR_INSTANTIATIONS(f, int, float) \
    BINARY_SCALAR_INSTANTIATIONS(f, int, int) \
    BINARY_SCALAR_INSTANTIATIONS(f, int, bool) \
    BINARY_SCALAR_INSTANTIATIONS(f, bool, double) \
    BINARY_SCALAR_INSTANTIATIONS(f, bool, float) \
    BINARY_SCALAR_INSTANTIATIONS(f, bool, int) \
    BINARY_SCALAR_INSTANTIATIONS(f, bool, bool)

namespace numbirch {

UNARY_TRANSFORM(operator+)
UNARY_TRANSFORM(operator-)
BINARY_TRANSFORM(operator+)
BINARY_TRANSFORM(operator-)
BINARY_SCALAR(operator*)
BINARY_SCALAR(operator/)
BINARY_COMPARE(operator==)
BINARY_COMPARE(operator!=)
BINARY_COMPARE(operator<)
BINARY_COMPARE(operator<=)
BINARY_COMPARE(operator>)
BINARY_COMPARE(operator>=)

BINARY_TRANSFORM(hadamard)

}
