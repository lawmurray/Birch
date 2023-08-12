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
 * @def ELEMENT_MATRIX
 * 
 * For single().
 */
#define ELEMENT_MATRIX(f) \
    ELEMENT_MATRIX_FIRST(f, real) \
    ELEMENT_MATRIX_FIRST(f, int) \
    ELEMENT_MATRIX_FIRST(f, bool)
#define ELEMENT_MATRIX_FIRST(f, T) \
    ELEMENT_MATRIX_SECOND(f, T, int)
#define ELEMENT_MATRIX_SECOND(f, T, U) \
    ELEMENT_MATRIX_THIRD(f, T, int, int)
#define ELEMENT_MATRIX_THIRD(f, T, U, V) \
    ELEMENT_MATRIX_SIG(f, T, U, V) \
    ELEMENT_MATRIX_SIG(f, T, U, NUMBIRCH_ARRAY(V, 0)) \
    ELEMENT_MATRIX_SIG(f, T, NUMBIRCH_ARRAY(U, 0), V) \
    ELEMENT_MATRIX_SIG(f, T, NUMBIRCH_ARRAY(U, 0), NUMBIRCH_ARRAY(V, 0))
#define ELEMENT_MATRIX_SIG(f, T, U, V) \
    template Array<T,0> f<T,U,V>(const Array<T,2>& x, const U& i, \
        const V& j); \
    template Array<real,2> f##_grad1(const Array<real,0>& g, \
        const Array<T,2>& A, const U& i, const V& j); \
    template real f##_grad2(const Array<real,0>& g, \
        const Array<T,2>& A, const U& i, const V& j); \
    template real f##_grad3(const Array<real,0>& g, \
        const Array<T,2>& A, const U& i, const V& j);

namespace numbirch {
ELEMENT_MATRIX(element)
}
