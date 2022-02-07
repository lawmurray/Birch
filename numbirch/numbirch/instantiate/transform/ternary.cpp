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

#define TERNARY(f, R) \
    TERNARY_FIRST(f, R, real) \
    TERNARY_FIRST(f, R, int) \
    TERNARY_FIRST(f, R, bool)
#define TERNARY_FIRST(f, R, T) \
    TERNARY_SECOND(f, R, T, real) \
    TERNARY_SECOND(f, R, T, int) \
    TERNARY_SECOND(f, R, T, bool)
#define TERNARY_SECOND(f, R, T, U) \
    TERNARY_THIRD(f, R, T, U, real) \
    TERNARY_THIRD(f, R, T, U, int) \
    TERNARY_THIRD(f, R, T, U, bool)
#define TERNARY_THIRD(f, R, T, U, V) \
    TERNARY_SIG(f, R, ARRAY(T, 2), ARRAY(U, 2), ARRAY(V, 2)) \
    TERNARY_SIG(f, R, ARRAY(T, 1), ARRAY(U, 1), ARRAY(V, 1)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0), ARRAY(V, 0)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0), V) \
    TERNARY_SIG(f, R, ARRAY(T, 0), U, ARRAY(V, 0)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), U, V) \
    TERNARY_SIG(f, R, T, ARRAY(U, 0), ARRAY(V, 0)) \
    TERNARY_SIG(f, R, T, ARRAY(U, 0), V) \
    TERNARY_SIG(f, R, T, U, ARRAY(V, 0)) \
    TERNARY_SIG(f, R, T, U, V) \
    TERNARY_SIG(f, R, ARRAY(T, 2), ARRAY(U, 2), ARRAY(V, 0)) \
    TERNARY_SIG(f, R, ARRAY(T, 2), ARRAY(U, 0), ARRAY(V, 2)) \
    TERNARY_SIG(f, R, ARRAY(T, 2), ARRAY(U, 0), ARRAY(V, 0)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 2), ARRAY(V, 2)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 2), ARRAY(V, 0)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0), ARRAY(V, 2)) \
    TERNARY_SIG(f, R, ARRAY(T, 1), ARRAY(U, 1), ARRAY(V, 0)) \
    TERNARY_SIG(f, R, ARRAY(T, 1), ARRAY(U, 0), ARRAY(V, 1)) \
    TERNARY_SIG(f, R, ARRAY(T, 1), ARRAY(U, 0), ARRAY(V, 0)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 1), ARRAY(V, 1)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 1), ARRAY(V, 0)) \
    TERNARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0), ARRAY(V, 1))
#define TERNARY_SIG(f, R, T, U, V) \
    template R<T,U,V> f<T,U,V,int>(const T&, const U&, const V&);

#define TERNARY_ARITHMETIC(f) TERNARY(f, implicit_t)
#define TERNARY_REAL(f) TERNARY(f, real_t)
#define TERNARY_BOOL(f) TERNARY(f, bool_t)

namespace numbirch {
TERNARY_REAL(ibeta)
}
