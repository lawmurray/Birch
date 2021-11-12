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
 * @def UNARY_ARITHMETIC
 * 
 * Explicitly instantiate a unary transformation `f` where the return type is
 * any arithmetic type.
 */
#define UNARY_ARITHMETIC(f) \
    UNARY_FIRST(f, double) \
    UNARY_FIRST(f, float) \
    UNARY_FIRST(f, int) \
    UNARY_FIRST(f, bool)
#define UNARY_RETURN(f) \
    UNARY_FIRST(f, double) \
    UNARY_FIRST(f, float)
#define UNARY_FIRST(f, R) \
    UNARY_DIM(f, R, double) \
    UNARY_DIM(f, R, float) \
    UNARY_DIM(f, R, int) \
    UNARY_DIM(f, R, bool)
#define UNARY_DIM(f, R, T) \
    UNARY_SIG(f, R, ARRAY(T, 2)) \
    UNARY_SIG(f, R, ARRAY(T, 1)) \
    UNARY_SIG(f, R, ARRAY(T, 0))
#define UNARY_SIG(f, R, T) \
    template convert_t<R,T> f<R>(const T&);

/**
 * @internal
 * 
 * @def UNARY_FLOATING_POINT
 * 
 * Explicitly instantiate a unary transformation `f` where the return type is
 * any floating point type.
 */
#define UNARY_FLOATING_POINT(f) \
    UNARY_FIRST(f, double) \
    UNARY_FIRST(f, float)

/**
 * @internal
 * 
 * @def UNARY_GRAD
 * 
 * Explicitly instantiate the gradient of a unary transformation `f`.
 */
#define UNARY_GRAD(f) \
    UNARY_GRAD_FIRST(f, double) \
    UNARY_GRAD_FIRST(f, float)
#define UNARY_GRAD_FIRST(f, G) \
    UNARY_GRAD_DIM(f, G, double) \
    UNARY_GRAD_DIM(f, G, float) \
    UNARY_GRAD_DIM(f, G, int) \
    UNARY_GRAD_DIM(f, G, bool)
#define UNARY_GRAD_DIM(f, G, T) \
    UNARY_GRAD_SIG(f, ARRAY(G, 2), ARRAY(T, 2)) \
    UNARY_GRAD_SIG(f, ARRAY(G, 1), ARRAY(T, 1)) \
    UNARY_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 0)) \
    UNARY_GRAD_SIG(f, ARRAY(G, 0), T) \
    UNARY_GRAD_SIG(f, G, ARRAY(T, 0))
#define UNARY_GRAD_SIG(f, G, T) \
    template promote_t<G,T> f<G,T>(const G&, const T&);

/**
 * @internal
 * 
 * @def UNARY_MATRIX
 * 
 * Explicitly instantiate a unary function `f` over floating point matrices.
 * Use cases include transpose(), inv().
 */
#define UNARY_MATRIX(f) \
    template Array<double,2> f(const Array<double,2>&); \
    template Array<float,2> f(const Array<float,2>&);

/**
 * @internal
 * 
 * @def UNARY_REDUCE_MATRIX
 * 
 * Explicitly instantiate a unary reduction function `f` for matrices of
 * floating point types only. Use cases include ldet(), trace().
 */
#define UNARY_REDUCE_MATRIX(f) \
    template Array<double,0> f(const Array<double,2>&); \
    template Array<float,0> f(const Array<float,2>&);

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
    BINARY_SIG(f, R, T, ARRAY(U, 0))
#define BINARY_SIG(f, R, T, U) \
    template convert_t<R,T,U> f<R,T,U>(const T&, const U&);

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

/**
 * @internal
 * 
 * @def BINARY_GRAD
 * 
 * Explicitly instantiate the gradient of a binary transformation `f`.
 */
#define BINARY_GRAD(f) \
    BINARY_GRAD_FIRST(f, double) \
    BINARY_GRAD_FIRST(f, float)
#define BINARY_GRAD_FIRST(f, G) \
    BINARY_GRAD_SECOND(f, G, double) \
    BINARY_GRAD_SECOND(f, G, float) \
    BINARY_GRAD_SECOND(f, G, int) \
    BINARY_GRAD_SECOND(f, G, bool)
#define BINARY_GRAD_SECOND(f, G, T) \
    BINARY_GRAD_DIM(f, G, T, double) \
    BINARY_GRAD_DIM(f, G, T, float) \
    BINARY_GRAD_DIM(f, G, T, int) \
    BINARY_GRAD_DIM(f, G, T, bool)
#define BINARY_GRAD_DIM(f, G, T, U) \
    BINARY_GRAD_SIG(f, ARRAY(G, 2), ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, ARRAY(G, 1), ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, ARRAY(G, 0), T, ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, G, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, G, ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, G, T, ARRAY(U, 0))
#define BINARY_GRAD_SIG(f, G, T, U) \
    template std::pair<promote_t<G,T,U>,promote_t<G,T,U>> f<G,T,U>(const G&, \
        const T&, const U&);

/**
 * @internal
 * 
 * @def BINARY_MATRIX_VECTOR
 * 
 * Explicitly instantiate a binary function `f` over a floating point matrix
 * and vector. Use cases include solve(), matrix-vector multiplication.
 */
#define BINARY_MATRIX_VECTOR(f) \
    BINARY_MATRIX_VECTOR_SIG(f, double) \
    BINARY_MATRIX_VECTOR_SIG(f, float)
#define BINARY_MATRIX_VECTOR_SIG(f, T) \
    template Array<T,1> f(const Array<T,2>&, const Array<T,1>&);

/**
 * @internal
 * 
 * @def BINARY_MATRIX_MATRIX
 * 
 * Explicitly instantiate a binary function `f` over floating point matrices.
 * Use cases include inner(), outer(), matrix-matrix multiplication.
 */
#define BINARY_MATRIX_MATRIX(f) \
    BINARY_MATRIX_MATRIX_SIG(f, double) \
    BINARY_MATRIX_MATRIX_SIG(f, float)
#define BINARY_MATRIX_MATRIX_SIG(f, T) \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&);

/**
 * @internal
 * 
 * @def BINARY_REDUCE_MATRIX
 * 
 * Explicitly instantiate a binary reduction function `f` for matrices of
 * floating point types only. Use cases include frobenius().
 */
#define BINARY_REDUCE_MATRIX(f) \
    template Array<double,0> f(const Array<double,2>&, const Array<double,2>&); \
    template Array<float,0> f(const Array<float,2>&, const Array<float,2>&);

/**
 * @internal
 * 
 * @def BINARY_SCALAR_MULTIPLY
 * 
 * Explicitly instantiate matrix-scalar multiplication.
 */
#define BINARY_SCALAR_MULTIPLY(f) \
    BINARY_SCALAR_DIVIDE_FIRST(f, double) \
    BINARY_SCALAR_DIVIDE_FIRST(f, float) \
    BINARY_SCALAR_DIVIDE_FIRST(f, int) \
    BINARY_SCALAR_DIVIDE_FIRST(f, bool)
#define BINARY_SCALAR_MULTIPLY_FIRST(f, R) \
    BINARY_SCALAR_MULTIPLY_SECOND(f, R, double) \
    BINARY_SCALAR_MULTIPLY_SECOND(f, R, float) \
    BINARY_SCALAR_MULTIPLY_SECOND(f, R, int) \
    BINARY_SCALAR_MULTIPLY_SECOND(f, R, bool)
#define BINARY_SCALAR_MULTIPLY_SECOND(f, R, T) \
    BINARY_SCALAR_MULTIPLY_LEFT(f, R, T, double) \
    BINARY_SCALAR_MULTIPLY_LEFT(f, R, T, float) \
    BINARY_SCALAR_MULTIPLY_LEFT(f, R, T, int) \
    BINARY_SCALAR_MULTIPLY_LEFT(f, R, T, bool) \
    BINARY_SCALAR_MULTIPLY_RIGHT(f, R, T, double) \
    BINARY_SCALAR_MULTIPLY_RIGHT(f, R, T, float) \
    BINARY_SCALAR_MULTIPLY_RIGHT(f, R, T, int) \
    BINARY_SCALAR_MULTIPLY_RIGHT(f, R, T, bool)
#define BINARY_SCALAR_MULTIPLY_LEFT(f, R, T, U) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 2)) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 1)) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, T, ARRAY(U, 2)) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, T, ARRAY(U, 1)) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, T, ARRAY(U, 0))
#define BINARY_SCALAR_MULTIPLY_RIGHT(f, R, T, U) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, ARRAY(T, 2), ARRAY(U, 0)) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, ARRAY(T, 1), ARRAY(U, 0)) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, ARRAY(T, 2), U) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, ARRAY(T, 1), U) \
    BINARY_SCALAR_MULTIPLY_SIG(f, R, ARRAY(T, 0), U)
#define BINARY_SCALAR_MULTIPLY_SIG(f, R, T, U) \
    template convert_t<R,T,U> f<R,T,U,int>(const T&, const U&);


/**
 * @internal
 * 
 * @def BINARY_SCALAR_DIVIDE
 * 
 * Explicitly instantiate matrix-scalar division.
 */
#define BINARY_SCALAR_DIVIDE(f) \
    BINARY_SCALAR_DIVIDE_FIRST(f, double) \
    BINARY_SCALAR_DIVIDE_FIRST(f, float) \
    BINARY_SCALAR_DIVIDE_FIRST(f, int) \
    BINARY_SCALAR_DIVIDE_FIRST(f, bool)
#define BINARY_SCALAR_DIVIDE_FIRST(f, R) \
    BINARY_SCALAR_DIVIDE_SECOND(f, R, double) \
    BINARY_SCALAR_DIVIDE_SECOND(f, R, float) \
    BINARY_SCALAR_DIVIDE_SECOND(f, R, int) \
    BINARY_SCALAR_DIVIDE_SECOND(f, R, bool)
#define BINARY_SCALAR_DIVIDE_SECOND(f, R, T) \
    BINARY_SCALAR_DIVIDE_RIGHT(f, R, T, double) \
    BINARY_SCALAR_DIVIDE_RIGHT(f, R, T, float) \
    BINARY_SCALAR_DIVIDE_RIGHT(f, R, T, int) \
    BINARY_SCALAR_DIVIDE_RIGHT(f, R, T, bool)
#define BINARY_SCALAR_DIVIDE_RIGHT(f, R, T, U) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 2), ARRAY(U, 0)) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 1), ARRAY(U, 0)) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 2), U) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 1), U) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, ARRAY(T, 0), U) \
    BINARY_SCALAR_DIVIDE_SIG(f, R, T, ARRAY(U, 0))
#define BINARY_SCALAR_DIVIDE_SIG(f, R, T, U) \
    template convert_t<R,T,U> f<R,T,U,int>(const T&, const U&);

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
    TERNARY_SIG(f, R, T, U, ARRAY(V, 0))
#define TERNARY_SIG(f, R, T, U, V) \
    template convert_t<R,T,U,V> f<R,T,U,V>(const T&, const U&, const V&);

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

UNARY_ARITHMETIC(operator+)
UNARY_ARITHMETIC(operator-)
BINARY_ARITHMETIC(operator+)
BINARY_ARITHMETIC(operator-)
BINARY_SCALAR_MULTIPLY(operator*)
BINARY_MATRIX_VECTOR(operator*)
BINARY_MATRIX_MATRIX(operator*)
BINARY_SCALAR_DIVIDE(operator/)
UNARY_ARITHMETIC(operator!)
BINARY_ARITHMETIC(operator&&)
BINARY_ARITHMETIC(operator||)
BINARY_ARITHMETIC(operator==)
BINARY_ARITHMETIC(operator!=)
BINARY_ARITHMETIC(operator<)
BINARY_ARITHMETIC(operator<=)
BINARY_ARITHMETIC(operator>)
BINARY_ARITHMETIC(operator>=)

UNARY_ARITHMETIC(abs)
UNARY_FLOATING_POINT(acos)
UNARY_FLOATING_POINT(asin)
UNARY_FLOATING_POINT(atan)
UNARY_ARITHMETIC(ceil)
UNARY_MATRIX(cholinv)
BINARY_MATRIX_VECTOR(cholmul)
BINARY_MATRIX_MATRIX(cholmul)
BINARY_MATRIX_MATRIX(cholouter)
BINARY_MATRIX_VECTOR(cholsolve)
BINARY_MATRIX_MATRIX(cholsolve)
BINARY_ARITHMETIC(copysign)
UNARY_FLOATING_POINT(cos)
UNARY_FLOATING_POINT(cosh)
COUNT(count)
DIAGONAL(diagonal)
UNARY_FLOATING_POINT(digamma)
BINARY_FLOATING_POINT(digamma)
DOT(dot)
UNARY_FLOATING_POINT(exp)
UNARY_FLOATING_POINT(expm1)
UNARY_ARITHMETIC(floor)
BINARY_REDUCE_MATRIX(frobenius)
BINARY_FLOATING_POINT(gamma_p)
BINARY_FLOATING_POINT(gamma_q)
BINARY_ARITHMETIC(hadamard)
TERNARY_FLOATING_POINT(ibeta)
BINARY_MATRIX_VECTOR(inner)
BINARY_MATRIX_MATRIX(inner)
UNARY_MATRIX(inv)
BINARY_FLOATING_POINT(lbeta)
UNARY_REDUCE_MATRIX(lcholdet)
BINARY_FLOATING_POINT(lchoose)
BINARY_GRAD(lchoose_grad)
UNARY_REDUCE_MATRIX(ldet)
UNARY_FLOATING_POINT(lfact)
UNARY_GRAD(lfact_grad)
UNARY_FLOATING_POINT(lgamma)
BINARY_FLOATING_POINT(lgamma)
UNARY_FLOATING_POINT(log)
UNARY_FLOATING_POINT(log1p)
OUTER(outer)
BINARY_MATRIX_MATRIX(outer)
BINARY_FLOATING_POINT(pow)
UNARY_FLOATING_POINT(rcp)
UNARY_ARITHMETIC(rectify)
UNARY_GRAD(rectify_grad)
UNARY_ARITHMETIC(round)
UNARY_FLOATING_POINT(sin)
SINGLE_VECTOR(single)
SINGLE_MATRIX(single)
UNARY_FLOATING_POINT(sinh)
BINARY_MATRIX_VECTOR(solve)
BINARY_MATRIX_MATRIX(solve)
UNARY_FLOATING_POINT(sqrt)
SUM(sum)
UNARY_FLOATING_POINT(tan)
UNARY_FLOATING_POINT(tanh)
UNARY_REDUCE_MATRIX(trace)
UNARY_MATRIX(transpose)

}
