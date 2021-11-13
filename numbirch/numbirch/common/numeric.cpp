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
 * @def MATRIX_VECTOR
 * 
 * Explicitly instantiate a binary function `f` over a floating point matrix
 * and vector. Use cases include solve(), matrix-vector multiplication.
 */
#define MATRIX_VECTOR(f) \
    MATRIX_VECTOR_SIG(f, double) \
    MATRIX_VECTOR_SIG(f, float)
#define MATRIX_VECTOR_SIG(f, T) \
    template Array<T,1> f(const Array<T,2>&, const Array<T,1>&);

/**
 * @internal
 * 
 * @def MATRIX_MATRIX
 * 
 * Explicitly instantiate a binary function `f` over floating point matrices.
 * Use cases include inner(), outer(), matrix-matrix multiplication.
 */
#define MATRIX_MATRIX(f) \
    MATRIX_MATRIX_SIG(f, double) \
    MATRIX_MATRIX_SIG(f, float)
#define MATRIX_MATRIX_SIG(f, T) \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&);

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
 * Explicitly instantiate dot().
 */
#define DOT(f) \
    DOT_SIG(f, double) \
    DOT_SIG(f, float)
#define DOT_SIG(f, T) \
    template Array<T,0> f(const Array<T,1>&, const Array<T,1>&);

/**
 * @internal
 * 
 * @def FROBENIUS
 * 
 * Explicitly instantiate frobenius().
 */
#define FROBENIUS(f) \
    FROBENIUS_SIG(f, double) \
    FROBENIUS_SIG(f, float)
#define FROBENIUS_SIG(f, T) \
    template Array<T,0> f(const Array<T,2>&, const Array<T,2>&);

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

/**
 * @internal
 * 
 * @def MATRIX
 * 
 * Explicitly instantiate a unary function `f` over floating point matrices.
 * Use cases include transpose(), inv().
 */
#define MATRIX(f) \
    template Array<double,2> f(const Array<double,2>&); \
    template Array<float,2> f(const Array<float,2>&);

/**
 * @internal
 * 
 * @def REDUCE_MATRIX
 * 
 * Explicitly instantiate a unary reduction function `f` for matrices of
 * floating point types only. Use cases include ldet(), trace().
 */
#define REDUCE_MATRIX(f) \
    template Array<double,0> f(const Array<double,2>&); \
    template Array<float,0> f(const Array<float,2>&);

namespace numbirch {
MATRIX_VECTOR(operator*)
MATRIX_MATRIX(operator*)
MATRIX(cholinv)
MATRIX_VECTOR(cholmul)
MATRIX_MATRIX(cholmul)
MATRIX_MATRIX(cholouter)
MATRIX_VECTOR(cholsolve)
MATRIX_MATRIX(cholsolve)
COUNT(count)
DIAGONAL(diagonal)
DOT(dot)
FROBENIUS(frobenius)
MATRIX_VECTOR(inner)
MATRIX_MATRIX(inner)
MATRIX(inv)
REDUCE_MATRIX(lcholdet)
REDUCE_MATRIX(ldet)
OUTER(outer)
MATRIX_MATRIX(outer)
MATRIX_VECTOR(solve)
MATRIX_MATRIX(solve)
SINGLE_VECTOR(single)
SINGLE_MATRIX(single)
SUM(sum)
REDUCE_MATRIX(trace)
MATRIX(transpose)

}
