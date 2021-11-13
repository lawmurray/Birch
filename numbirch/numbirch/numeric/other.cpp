/**
 * @file
 */
#include "numbirch/numeric.hpp"

#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.hpp"
#endif

/**
 * @internal
 * 
 * @def SINGLE_VECTOR
 * 
 * For single().
 */
#define SINGLE_VECTOR(f) \
    SINGLE_VECTOR_FIRST(f, double) \
    SINGLE_VECTOR_FIRST(f, float) \
    SINGLE_VECTOR_FIRST(f, int) \
    SINGLE_VECTOR_FIRST(f, bool)
#define SINGLE_VECTOR_FIRST(f, R) \
    SINGLE_VECTOR_SIG(f, R, int) \
    SINGLE_VECTOR_SIG(f, R, ARRAY(int, 0))
#define SINGLE_VECTOR_SIG(f, R, T) \
    template Array<R,1> f<R>(const T& i, const int n);

/**
 * @internal
 * 
 * @def SINGLE_MATRIX
 * 
 * For single().
 */
#define SINGLE_MATRIX(f) \
    SINGLE_MATRIX_FIRST(f, double) \
    SINGLE_MATRIX_FIRST(f, float) \
    SINGLE_MATRIX_FIRST(f, int) \
    SINGLE_MATRIX_FIRST(f, bool)
#define SINGLE_MATRIX_FIRST(f, R) \
    SINGLE_MATRIX_SIG(f, R, ARRAY(int, 0), ARRAY(int, 0)) \
    SINGLE_MATRIX_SIG(f, R, ARRAY(int, 0), int) \
    SINGLE_MATRIX_SIG(f, R, int, ARRAY(int, 0)) \
    SINGLE_MATRIX_SIG(f, R, int, int)
#define SINGLE_MATRIX_SIG(f, R, T, U) \
    template Array<R,2> f<R>(const T& i, const U& j, const int m, const int n);

/**
 * @internal
 * 
 * @def DIAGONAL
 * 
 * Explicitly instantiate diagonal().
 */
#define DIAGONAL(f) \
    DIAGONAL_FIRST(f, double) \
    DIAGONAL_FIRST(f, float) \
    DIAGONAL_FIRST(f, int) \
    DIAGONAL_FIRST(f, bool)
#define DIAGONAL_FIRST(f, R) \
    DIAGONAL_DIM(f, R, double) \
    DIAGONAL_DIM(f, R, float) \
    DIAGONAL_DIM(f, R, int) \
    DIAGONAL_DIM(f, R, bool)
#define DIAGONAL_DIM(f, R, T) \
    DIAGONAL_SIG(f, R, T) \
    DIAGONAL_SIG(f, R, ARRAY(T, 0))
#define DIAGONAL_SIG(f, R, T) \
    template Array<R,2> f<R,T,int>(const T& x, const int n);

/**
 * @internal
 * 
 * @def SUM
 * 
 * Explicitly instantiate sum().
 */
#define SUM(f) \
    SUM_FIRST(f, double) \
    SUM_FIRST(f, float) \
    SUM_FIRST(f, int) \
    SUM_FIRST(f, bool)
#define SUM_FIRST(f, R) \
    SUM_DIM(f, R, double) \
    SUM_DIM(f, R, float) \
    SUM_DIM(f, R, int) \
    SUM_DIM(f, R, bool)
#define SUM_DIM(f, R, T) \
    SUM_SIG(f, R, ARRAY(T, 2)) \
    SUM_SIG(f, R, ARRAY(T, 1)) \
    SUM_SIG(f, R, ARRAY(T, 0))
#define SUM_SIG(f, R, T) \
    template Array<R,0> f<R>(const T&);

/**
 * @internal
 * 
 * @def COUNT
 * 
 * Explicitly instantiate count().
 */
#define COUNT(f) \
    COUNT_FIRST(f, double) \
    COUNT_FIRST(f, float) \
    COUNT_FIRST(f, int) \
    COUNT_FIRST(f, bool)
#define COUNT_FIRST(f, R) \
    COUNT_DIM(f, R, double) \
    COUNT_DIM(f, R, float) \
    COUNT_DIM(f, R, int) \
    COUNT_DIM(f, R, bool)
#define COUNT_DIM(f, R, T) \
    COUNT_SIG(f, R, ARRAY(T, 2)) \
    COUNT_SIG(f, R, ARRAY(T, 1)) \
    COUNT_SIG(f, R, ARRAY(T, 0))
#define COUNT_SIG(f, R, T) \
    template Array<R,0> f<R>(const T&);

/**
 * @internal
 * 
 * @def DOT
 * 
 * Explicitly instantiate dot product of vectors.
 */
#define DOT(f) \
    DOT_SIG(f, double) \
    DOT_SIG(f, float)
#define DOT_SIG(f, T) \
    template Array<T,0> f(const Array<T,1>&, const Array<T,1>&);

/**
 * @internal
 * 
 * @def OUTER
 * 
 * Explicitly instantiate outer product of vectors.
 */
#define OUTER(f) \
    OUTER_SIG(f, double) \
    OUTER_SIG(f, float)
#define OUTER_SIG(f, T) \
    template Array<T,2> f(const Array<T,1>&, const Array<T,1>&);

namespace numbirch {

COUNT(count)
DIAGONAL(diagonal)
DOT(dot)
OUTER(outer)
SINGLE_VECTOR(single)
SINGLE_MATRIX(single)
SUM(sum)

}
