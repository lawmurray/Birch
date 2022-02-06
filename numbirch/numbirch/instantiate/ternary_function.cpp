/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/ternary.hpp"

#define TERNARY_ARITHMETIC(f) \
    TERNARY_FIRST(f, real) \
    TERNARY_FIRST(f, int) \
    TERNARY_FIRST(f, bool)
#define TERNARY_FIRST(f, T) \
    TERNARY_SECOND(f, T, real) \
    TERNARY_SECOND(f, T, int) \
    TERNARY_SECOND(f, T, bool)
#define TERNARY_SECOND(f, T, U) \
    TERNARY_THIRD(f, T, U, real) \
    TERNARY_THIRD(f, T, U, int) \
    TERNARY_THIRD(f, T, U, bool)
#define TERNARY_THIRD(f, T, U, V) \
    TERNARY_SIG(f, ARRAY(T, 2), ARRAY(U, 2), ARRAY(V, 2)) \
    TERNARY_SIG(f, ARRAY(T, 1), ARRAY(U, 1), ARRAY(V, 1)) \
    TERNARY_SIG(f, ARRAY(T, 0), ARRAY(U, 0), ARRAY(V, 0)) \
    TERNARY_SIG(f, ARRAY(T, 0), ARRAY(U, 0), V) \
    TERNARY_SIG(f, ARRAY(T, 0), U, ARRAY(V, 0)) \
    TERNARY_SIG(f, ARRAY(T, 0), U, V) \
    TERNARY_SIG(f, T, ARRAY(U, 0), ARRAY(V, 0)) \
    TERNARY_SIG(f, T, ARRAY(U, 0), V) \
    TERNARY_SIG(f, T, U, ARRAY(V, 0)) \
    TERNARY_SIG(f, T, U, V)
#define TERNARY_SIG(f, T, U, V) \
    template default_t<T,U,V> f<T,U,V,int>(const T&, const U&, const V&);

#define TERNARY_FLOATING_POINT(f) \
    TERNARY_FIRST(f, real)

namespace numbirch {
TERNARY_FLOATING_POINT(ibeta)
}
