/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#endif
#include "numbirch/common/array.inl"

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
    SINGLE_MATRIX_SIG(f, T, U, NUMBIRCH_ARRAY(V, 0)) \
    SINGLE_MATRIX_SIG(f, T, NUMBIRCH_ARRAY(U, 0), V) \
    SINGLE_MATRIX_SIG(f, T, NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 0)) \
    SINGLE_MATRIX_SIG(f, NUMBIRCH_ARRAY(T, 0), U, V) \
    SINGLE_MATRIX_SIG(f, NUMBIRCH_ARRAY(T, 0), U, NUMBIRCH_ARRAY(V, 0)) \
    SINGLE_MATRIX_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0), V) \
    SINGLE_MATRIX_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 0))
#define SINGLE_MATRIX_SIG(f, T, U, V) \
    template Array<value_t<T>,2> f<T,U,V,int>(const T& x, const U& i, \
        const V& j, const int m, const int n); \
    template Array<real,0> f##_grad1(const Array<real,2>& g, \
        const Array<value_t<T>,2>& A, const T& x, const U& i, const V& j, \
        const int m, const int n); \
    template real f##_grad2(const Array<real,2>& g, \
        const Array<value_t<T>,2>& A, const T& x, const U& i, const V& j, \
        const int m, const int n); \
    template real f##_grad3(const Array<real,2>& g, \
        const Array<value_t<T>,2>& A, const T& x, const U& i, const V& j, \
        const int m, const int n);

namespace numbirch {
SINGLE_MATRIX(single)
}
