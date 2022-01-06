/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/array.hpp"

/**
 * @internal
 * 
 * @def SINGLE_MATRIX
 * 
 * For single().
 */
#define SINGLE_MATRIX(f) \
    SINGLE_MATRIX_FIRST(f, real) \
    SINGLE_MATRIX_FIRST(f, int) \
    SINGLE_MATRIX_FIRST(f, bool)
#define SINGLE_MATRIX_FIRST(f, T) \
    SINGLE_MATRIX_SECOND(f, T, int)
#define SINGLE_MATRIX_SECOND(f, T, U) \
    SINGLE_MATRIX_THIRD(f, T, int, int)
#define SINGLE_MATRIX_THIRD(f, T, U, V) \
    SINGLE_MATRIX_SIG(f, T, U, V) \
    SINGLE_MATRIX_SIG(f, T, U, ARRAY(V, 0)) \
    SINGLE_MATRIX_SIG(f, T, ARRAY(U, 0), V) \
    SINGLE_MATRIX_SIG(f, T, ARRAY(U, 0), ARRAY(V, 0)) \
    SINGLE_MATRIX_SIG(f, ARRAY(T, 0), U, V) \
    SINGLE_MATRIX_SIG(f, ARRAY(T, 0), U, ARRAY(V, 0)) \
    SINGLE_MATRIX_SIG(f, ARRAY(T, 0), ARRAY(U, 0), V) \
    SINGLE_MATRIX_SIG(f, ARRAY(T, 0), ARRAY(U, 0), ARRAY(V, 0))
#define SINGLE_MATRIX_SIG(f, T, U, V) \
    template Array<value_t<T>,2> f<T,U,V,int>(const T& x, const U& i, \
        const V& j, const int m, const int n);

namespace numbirch {
SINGLE_MATRIX(single)
}
