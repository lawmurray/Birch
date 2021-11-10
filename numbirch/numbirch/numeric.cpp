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

#define UNARY_INSTANTIATION(f, T) \
    template T f(const T&);
#define UNARY_INSTANTIATIONS(f, T) \
    UNARY_INSTANTIATION(f, ARRAY(T, 2)) \
    UNARY_INSTANTIATION(f, ARRAY(T, 1)) \
    UNARY_INSTANTIATION(f, ARRAY(T, 0))

#define UNARY_EXPLICIT_INSTANTIATION(f, T, U) \
    template promote_t<T,U> f<T>(const U&);
#define UNARY_EXPLICIT_INSTANTIATIONS(f, T, U) \
    UNARY_EXPLICIT_INSTANTIATION(f, T, ARRAY(U, 2)) \
    UNARY_EXPLICIT_INSTANTIATION(f, T, ARRAY(U, 1)) \
    UNARY_EXPLICIT_INSTANTIATION(f, T, ARRAY(U, 0))

/**
 * @internal
 * 
 * @def UNARY
 * 
 * Explicitly instantiate a unary transformation `f` for all types.
 */
#define UNARY(f) \
    UNARY_INSTANTIATIONS(f, double) \
    UNARY_INSTANTIATIONS(f, float) \
    UNARY_INSTANTIATIONS(f, int) \
    UNARY_INSTANTIATIONS(f, bool)

/**
 * @internal
 * 
 * @def UNARY_EXPLICIT
 * 
 * Explicitly instantiate a unary transformation `f` for all types and sizes,
 * where the result type must be explicitly specified in the case of integral
 * arguments.
 * 
 * For example, `int abs(int)` is *not* in this category, as the result type
 * of `abs` is always the same as the argument type, while `double sin(int)`
 * *is* in this category, as the return type must be explicitly specified in
 * the case of an integral argument.
 */
#define UNARY_EXPLICIT(f) \
    UNARY_INSTANTIATIONS(f, double) \
    UNARY_INSTANTIATIONS(f, float) \
    UNARY_EXPLICIT_INSTANTIATIONS(f, double, int) \
    UNARY_EXPLICIT_INSTANTIATIONS(f, double, bool) \
    UNARY_EXPLICIT_INSTANTIATIONS(f, float, int) \
    UNARY_EXPLICIT_INSTANTIATIONS(f, float, bool)

/**
 * @internal
 * 
 * @def UNARY_GRAD
 * 
 * Explicitly instantiate a unary gradient `f` for all types.
 */
#define UNARY_GRAD(f) \
    BINARY_SECOND(f, double) \
    BINARY_SECOND(f, float)

/**
 * @internal
 * 
 * @def UNARY_MATRIX
 * 
 * Explicitly instantiate a unary function `f` for a matrix for floating point
 * types only. Use cases include inv(), cholinv(), transpose().
 */
#define UNARY_MATRIX(f) \
    template Array<double,2> f(const Array<double,2>&); \
    template Array<float,2> f(const Array<float,2>&);

/**
 * @internal
 * 
 * @def REDUCE_MATRIX
 * 
 * Explicitly instantiate a reduction function `f` for a matrix for floating
 * point types only. Use cases include ldet(), lcholdet(), trace().
 */
#define REDUCE_MATRIX(f) \
    template Array<double,0> f(const Array<double,2>&); \
    template Array<float,0> f(const Array<float,2>&);

/**
 * @internal
 * 
 * @def BINARY
 * 
 * Explicitly instantiate a binary transformation `f` for all pairs of
 * compatible types.
 */
#define BINARY_SIG(f, T, U) \
    template promote_t<T,U> f(const T&, const U&);
#define BINARY_DIM(f, T, U) \
    BINARY_SIG(f, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_SIG(f, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_SIG(f, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_SIG(f, ARRAY(T, 0), U) \
    BINARY_SIG(f, T, ARRAY(U, 0))
#define BINARY_SECOND(f, T) \
    BINARY_DIM(f, T, double) \
    BINARY_DIM(f, T, float) \
    BINARY_DIM(f, T, int) \
    BINARY_DIM(f, T, bool)
#define BINARY_FIRST(f) \
    BINARY_SECOND(f, double) \
    BINARY_SECOND(f, float) \
    BINARY_SECOND(f, int) \
    BINARY_SECOND(f, bool)
#define BINARY(f) \
    BINARY_FIRST(f)

/**
 * @internal
 * 
 * @def COMPARE
 * 
 * Explicitly instantiate a binary comparison `f` for all pairs of
 * compatible types.
 */
#define COMPARE_SIG(f, T, U) \
    template Array<bool,dimension_v<T>> f(const T&, const U&);
#define COMPARE_DIM(f, T, U) \
    COMPARE_SIG(f, ARRAY(T, 2), ARRAY(U, 2)) \
    COMPARE_SIG(f, ARRAY(T, 1), ARRAY(U, 1)) \
    COMPARE_SIG(f, ARRAY(T, 0), ARRAY(U, 0)) \
    COMPARE_SIG(f, ARRAY(T, 0), U) \
    COMPARE_SIG(f, T, ARRAY(U, 0))
#define COMPARE_SECOND(f, T) \
    COMPARE_DIM(f, T, double) \
    COMPARE_DIM(f, T, float) \
    COMPARE_DIM(f, T, int) \
    COMPARE_DIM(f, T, bool)
#define COMPARE_FIRST(f) \
    COMPARE_SECOND(f, double) \
    COMPARE_SECOND(f, float) \
    COMPARE_SECOND(f, int) \
    COMPARE_SECOND(f, bool)
#define COMPARE(f) \
    COMPARE_FIRST(f)

/**
 * @internal
 * 
 * @def BINARY_REDUCE
 * 
 * Explicitly instantiate a binary transform-and-reduce `f` for all pairs of
 * compatible types.
 */
#define BINARY_REDUCE_SIG(f, T, U) \
    template Array<value_t<promote_t<T,U>>,0> f(const T&, const U&);
#define BINARY_REDUCE_DIM(f, T, U) \
    BINARY_REDUCE_SIG(f, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_REDUCE_SIG(f, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_REDUCE_SIG(f, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_REDUCE_SIG(f, ARRAY(T, 0), U) \
    BINARY_REDUCE_SIG(f, T, ARRAY(U, 0))
#define BINARY_REDUCE_SECOND(f, T) \
    BINARY_REDUCE_DIM(f, T, double) \
    BINARY_REDUCE_DIM(f, T, float) \
    BINARY_REDUCE_DIM(f, T, int) \
    BINARY_REDUCE_DIM(f, T, bool)
#define BINARY_REDUCE_FIRST(f) \
    BINARY_REDUCE_SECOND(f, double) \
    BINARY_REDUCE_SECOND(f, float) \
    BINARY_REDUCE_SECOND(f, int) \
    BINARY_REDUCE_SECOND(f, bool)
#define BINARY_REDUCE(f) \
    BINARY_REDUCE_FIRST(f)

/**
 * @internal
 * 
 * @def LEFT_SCALAR
 * 
 * Explicitly instantiate a binary function `f` with a scalar as the right
 * argument, for all pairs of compatible types. Excludes scalar-scalar
 * instantiations, as if these are required `RIGHT_SCALAR` will include them.
 */
#define LEFT_SCALAR_INSTANTIATION(f, T, U) \
    template promote_t<T,U> f(const T&, const U&);
#define LEFT_SCALAR_INSTANTIATIONS(f, T, U) \
    LEFT_SCALAR_INSTANTIATION(f, ARRAY(T, 0), ARRAY(U, 2)) \
    LEFT_SCALAR_INSTANTIATION(f, T, ARRAY(U, 2)) \
    LEFT_SCALAR_INSTANTIATION(f, ARRAY(T, 0), ARRAY(U, 1)) \
    LEFT_SCALAR_INSTANTIATION(f, T, ARRAY(U, 1))
#define LEFT_SCALAR(f) \
    LEFT_SCALAR_INSTANTIATIONS(f, double, double) \
    LEFT_SCALAR_INSTANTIATIONS(f, double, float) \
    LEFT_SCALAR_INSTANTIATIONS(f, double, int) \
    LEFT_SCALAR_INSTANTIATIONS(f, double, bool) \
    LEFT_SCALAR_INSTANTIATIONS(f, float, double) \
    LEFT_SCALAR_INSTANTIATIONS(f, float, float) \
    LEFT_SCALAR_INSTANTIATIONS(f, float, int) \
    LEFT_SCALAR_INSTANTIATIONS(f, float, bool) \
    LEFT_SCALAR_INSTANTIATIONS(f, int, double) \
    LEFT_SCALAR_INSTANTIATIONS(f, int, float) \
    LEFT_SCALAR_INSTANTIATIONS(f, int, int) \
    LEFT_SCALAR_INSTANTIATIONS(f, int, bool) \
    LEFT_SCALAR_INSTANTIATIONS(f, bool, double) \
    LEFT_SCALAR_INSTANTIATIONS(f, bool, float) \
    LEFT_SCALAR_INSTANTIATIONS(f, bool, int) \
    LEFT_SCALAR_INSTANTIATIONS(f, bool, bool)

/**
 * @internal
 * 
 * @def RIGHT_SCALAR
 * 
 * Explicitly instantiate a binary function `f` with a scalar as the right
 * argument, for all pairs of compatible types.
 */
#define RIGHT_SCALAR_INSTANTIATION(f, T, U) \
    template promote_t<T,U> f(const T&, const U&);
#define RIGHT_SCALAR_INSTANTIATIONS(f, T, U) \
    RIGHT_SCALAR_INSTANTIATION(f, ARRAY(T, 2), ARRAY(U, 0)) \
    RIGHT_SCALAR_INSTANTIATION(f, ARRAY(T, 2), U) \
    RIGHT_SCALAR_INSTANTIATION(f, ARRAY(T, 1), ARRAY(U, 0)) \
    RIGHT_SCALAR_INSTANTIATION(f, ARRAY(T, 1), U) \
    RIGHT_SCALAR_INSTANTIATION(f, ARRAY(T, 0), ARRAY(U, 0)) \
    RIGHT_SCALAR_INSTANTIATION(f, ARRAY(T, 0), U) \
    RIGHT_SCALAR_INSTANTIATION(f, T, ARRAY(U, 0))
#define RIGHT_SCALAR(f) \
    RIGHT_SCALAR_INSTANTIATIONS(f, double, double) \
    RIGHT_SCALAR_INSTANTIATIONS(f, double, float) \
    RIGHT_SCALAR_INSTANTIATIONS(f, double, int) \
    RIGHT_SCALAR_INSTANTIATIONS(f, double, bool) \
    RIGHT_SCALAR_INSTANTIATIONS(f, float, double) \
    RIGHT_SCALAR_INSTANTIATIONS(f, float, float) \
    RIGHT_SCALAR_INSTANTIATIONS(f, float, int) \
    RIGHT_SCALAR_INSTANTIATIONS(f, float, bool) \
    RIGHT_SCALAR_INSTANTIATIONS(f, int, double) \
    RIGHT_SCALAR_INSTANTIATIONS(f, int, float) \
    RIGHT_SCALAR_INSTANTIATIONS(f, int, int) \
    RIGHT_SCALAR_INSTANTIATIONS(f, int, bool) \
    RIGHT_SCALAR_INSTANTIATIONS(f, bool, double) \
    RIGHT_SCALAR_INSTANTIATIONS(f, bool, float) \
    RIGHT_SCALAR_INSTANTIATIONS(f, bool, int) \
    RIGHT_SCALAR_INSTANTIATIONS(f, bool, bool)

/**
 * @internal
 * 
 * @def BINARY_EXPLICIT
 * 
 * Explicitly instantiate a binary transformation `f` for all types and sizes,
 * where the result type must be explicitly specified in the case of integral
 * arguments.
 */
#define BINARY_EXPLICIT_SIG(f, T, U, V) \
    template promote_t<T,U> f<T>(const U&, const V&);
#define BINARY_EXPLICIT_DIM(f, T, U, V) \
    BINARY_EXPLICIT_SIG(f, T, ARRAY(U, 2), ARRAY(V, 2)) \
    BINARY_EXPLICIT_SIG(f, T, ARRAY(U, 1), ARRAY(V, 1)) \
    BINARY_EXPLICIT_SIG(f, T, ARRAY(U, 0), ARRAY(V, 0)) \
    BINARY_EXPLICIT_SIG(f, T, ARRAY(U, 0), V) \
    BINARY_EXPLICIT_SIG(f, T, U, ARRAY(V, 0))
#define BINARY_EXPLICIT_THIRD(f, T, U) \
    BINARY_EXPLICIT_DIM(f, T, U, int) \
    BINARY_EXPLICIT_DIM(f, T, U, bool)
#define BINARY_EXPLICIT_SECOND(f, T) \
    BINARY_EXPLICIT_THIRD(f, T, int) \
    BINARY_EXPLICIT_THIRD(f, T, bool)
#define BINARY_EXPLICIT_FIRST(f) \
    BINARY_EXPLICIT_SECOND(f, double) \
    BINARY_EXPLICIT_SECOND(f, float)
#define BINARY_EXPLICIT(f) \
    BINARY_EXPLICIT_FIRST(f) \
    BINARY_SECOND(f, double) \
    BINARY_SECOND(f, float) \
    BINARY_DIM(f, int, double) \
    BINARY_DIM(f, int, float) \
    BINARY_DIM(f, bool, double) \
    BINARY_DIM(f, bool, float)

/**
 * @internal
 * 
 * @def SINGLE_VECTOR
 * 
 * For single().
 */
#define SINGLE_VECTOR_INSTANTIATION(f, T, U) \
    template Array<T,1> f(const U& i, const int n);
#define SINGLE_VECTOR_INSTANTIATIONS(f, T) \
    SINGLE_VECTOR_INSTANTIATION(f, T, int) \
    SINGLE_VECTOR_INSTANTIATION(f, T, ARRAY(int, 0))
#define SINGLE_VECTOR(f) \
    SINGLE_VECTOR_INSTANTIATIONS(f, double) \
    SINGLE_VECTOR_INSTANTIATIONS(f, float) \
    SINGLE_VECTOR_INSTANTIATIONS(f, int) \
    SINGLE_VECTOR_INSTANTIATIONS(f, bool)

/**
 * @internal
 * 
 * @def SINGLE_MATRIX
 * 
 * For single().
 */
#define SINGLE_MATRIX_INSTANTIATION(f, T, U, V) \
    template Array<T,2> f(const U& i, const V& j, const int m, const int n);
#define SINGLE_MATRIX_INSTANTIATIONS(f, T) \
    SINGLE_MATRIX_INSTANTIATION(f, T, ARRAY(int, 0), ARRAY(int, 0)) \
    SINGLE_MATRIX_INSTANTIATION(f, T, ARRAY(int, 0), int) \
    SINGLE_MATRIX_INSTANTIATION(f, T, int, ARRAY(int, 0)) \
    SINGLE_MATRIX_INSTANTIATION(f, T, int, int)
#define SINGLE_MATRIX(f) \
    SINGLE_MATRIX_INSTANTIATIONS(f, double) \
    SINGLE_MATRIX_INSTANTIATIONS(f, float) \
    SINGLE_MATRIX_INSTANTIATIONS(f, int) \
    SINGLE_MATRIX_INSTANTIATIONS(f, bool)

/**
 * @internal
 * 
 * @def DIAGONAL
 * 
 * For diagonal().
 */
#define DIAGONAL_INSTANTIATION(f, T) \
    template Array<value_t<T>,2> f(const T& x, const int n);
#define DIAGONAL_INSTANTIATIONS(f, T) \
    DIAGONAL_INSTANTIATION(f, T) \
    DIAGONAL_INSTANTIATION(f, ARRAY(T, 0))
#define DIAGONAL(f) \
    DIAGONAL_INSTANTIATIONS(f, double) \
    DIAGONAL_INSTANTIATIONS(f, float) \
    DIAGONAL_INSTANTIATIONS(f, int) \
    DIAGONAL_INSTANTIATIONS(f, bool)

/**
 * @internal
 * 
 * @def SUM
 * 
 * Explicitly instantiate a reduction function `f`. Archetype is sum().
 */
#define SUM_INSTANTIATION(f, T) \
    template Array<value_t<T>,0> f(const T&);
#define SUM_INSTANTIATIONS(f, T) \
    SUM_INSTANTIATION(f, ARRAY(T, 2)) \
    SUM_INSTANTIATION(f, ARRAY(T, 1)) \
    SUM_INSTANTIATION(f, ARRAY(T, 0))
#define SUM(f) \
    SUM_INSTANTIATIONS(f, double) \
    SUM_INSTANTIATIONS(f, float) \
    SUM_INSTANTIATIONS(f, int) \
    SUM_INSTANTIATIONS(f, bool)

/**
 * @internal
 * 
 * @def COUNT
 * 
 * Explicitly instantiate a reduction function `f` with integral return type.
 * Archetype is count().
 */
#define COUNT_INSTANTIATION(f, T) \
    template Array<int,0> f(const T&);
#define COUNT_INSTANTIATIONS(f, T) \
    COUNT_INSTANTIATION(f, ARRAY(T, 2)) \
    COUNT_INSTANTIATION(f, ARRAY(T, 1)) \
    COUNT_INSTANTIATION(f, ARRAY(T, 0))
#define COUNT(f) \
    COUNT_INSTANTIATIONS(f, double) \
    COUNT_INSTANTIATIONS(f, float) \
    COUNT_INSTANTIATIONS(f, int) \
    COUNT_INSTANTIATIONS(f, bool)

/**
 * @internal
 * 
 * @def DOT
 * 
 * Explicitly instantiate dot product of vectors.
 */
#define DOT_SIG(f, T) \
    template Array<T,0> f(const Array<T,1>&, const Array<T,1>&);
#define DOT(f) \
    DOT_SIG(f, double) \
    DOT_SIG(f, float)

/**
 * @internal
 * 
 * @def OUTER
 * 
 * Explicitly instantiate outer product of vectors.
 */
#define OUTER_SIG(f, T) \
    template Array<T,2> f(const Array<T,1>&, const Array<T,1>&);
#define OUTER(f) \
    OUTER_SIG(f, double) \
    OUTER_SIG(f, float)

/**
 * @internal
 * 
 * @def MATRIX_VECTOR
 * 
 * Explicitly instantiate matrix multiplications.
 */
#define MATRIX_VECTOR_INSTANTIATIONS(f, T) \
    template Array<T,1> f(const Array<T,2>&, const Array<T,1>&);
#define MATRIX_VECTOR(f) \
    MATRIX_VECTOR_INSTANTIATIONS(f, double) \
    MATRIX_VECTOR_INSTANTIATIONS(f, float)

/**
 * @internal
 * 
 * @def MATRIX_MATRIX
 * 
 * Explicitly instantiate matrix multiplications.
 */
#define MATRIX_MATRIX_INSTANTIATIONS(f, T) \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&);
#define MATRIX_MATRIX(f) \
    MATRIX_MATRIX_INSTANTIATIONS(f, double) \
    MATRIX_MATRIX_INSTANTIATIONS(f, float)

namespace numbirch {

UNARY(operator+)
UNARY(operator-)
BINARY(operator+)
BINARY(operator-)
LEFT_SCALAR(operator*)
RIGHT_SCALAR(operator*)
MATRIX_VECTOR(operator*)
MATRIX_MATRIX(operator*)
RIGHT_SCALAR(operator/)
UNARY(operator!)
COMPARE(operator&&)
COMPARE(operator||)
COMPARE(operator==)
COMPARE(operator!=)
COMPARE(operator<)
COMPARE(operator<=)
COMPARE(operator>)
COMPARE(operator>=)

UNARY(abs)
UNARY_EXPLICIT(acos)
UNARY_EXPLICIT(asin)
UNARY_EXPLICIT(atan)
UNARY(ceil)
UNARY_MATRIX(cholinv)
MATRIX_VECTOR(cholmul)
MATRIX_MATRIX(cholmul)
MATRIX_MATRIX(cholouter)
MATRIX_VECTOR(cholsolve)
MATRIX_MATRIX(cholsolve)
BINARY(copysign)
UNARY_EXPLICIT(cos)
UNARY_EXPLICIT(cosh)
COUNT(count)
DIAGONAL(diagonal)
UNARY_EXPLICIT(digamma)
BINARY_EXPLICIT(digamma)
DOT(dot)
UNARY_EXPLICIT(exp)
UNARY_EXPLICIT(expm1)
UNARY(floor)
BINARY_REDUCE(frobenius)
BINARY_EXPLICIT(gamma_p)
BINARY_EXPLICIT(gamma_q)
BINARY(hadamard)
//TERNARY_EXPLICIT(ibeta)
MATRIX_VECTOR(inner)
MATRIX_MATRIX(inner)
UNARY_MATRIX(inv)
BINARY_EXPLICIT(lbeta)
REDUCE_MATRIX(lcholdet)
BINARY_EXPLICIT(lchoose)
REDUCE_MATRIX(ldet)
UNARY_EXPLICIT(lfact)
UNARY_GRAD(lfact_grad)
UNARY_EXPLICIT(lgamma)
BINARY_EXPLICIT(lgamma)
UNARY_EXPLICIT(log)
UNARY_EXPLICIT(log1p)
OUTER(outer)
MATRIX_MATRIX(outer)
BINARY_EXPLICIT(pow)
UNARY_EXPLICIT(rcp)
UNARY(rectify)
UNARY_GRAD(rectify_grad)
UNARY(round)
UNARY_EXPLICIT(sin)
SINGLE_VECTOR(single)
SINGLE_MATRIX(single)
UNARY_EXPLICIT(sinh)
MATRIX_VECTOR(solve)
MATRIX_MATRIX(solve)
UNARY_EXPLICIT(sqrt)
SUM(sum)
UNARY_EXPLICIT(tan)
UNARY_EXPLICIT(tanh)
REDUCE_MATRIX(trace)
UNARY_MATRIX(transpose)

}
