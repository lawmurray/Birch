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

namespace numbirch {
UNARY_ARITHMETIC(operator+)
UNARY_ARITHMETIC(operator-)
UNARY_ARITHMETIC(operator!)

UNARY_ARITHMETIC(abs)
UNARY_FLOATING_POINT(acos)
UNARY_FLOATING_POINT(asin)
UNARY_FLOATING_POINT(atan)
UNARY_ARITHMETIC(ceil)
UNARY_MATRIX(cholinv)
UNARY_FLOATING_POINT(cos)
UNARY_FLOATING_POINT(cosh)
UNARY_FLOATING_POINT(digamma)
UNARY_FLOATING_POINT(exp)
UNARY_FLOATING_POINT(expm1)
UNARY_ARITHMETIC(floor)
UNARY_MATRIX(inv)
UNARY_REDUCE_MATRIX(lcholdet)
UNARY_REDUCE_MATRIX(ldet)
UNARY_FLOATING_POINT(lfact)
UNARY_GRAD(lfact_grad)
UNARY_FLOATING_POINT(lgamma)
UNARY_FLOATING_POINT(log)
UNARY_FLOATING_POINT(log1p)
UNARY_FLOATING_POINT(rcp)
UNARY_ARITHMETIC(rectify)
UNARY_GRAD(rectify_grad)
UNARY_ARITHMETIC(round)
UNARY_FLOATING_POINT(sin)
UNARY_FLOATING_POINT(sinh)
UNARY_FLOATING_POINT(sqrt)
UNARY_FLOATING_POINT(tan)
UNARY_FLOATING_POINT(tanh)
UNARY_REDUCE_MATRIX(trace)
UNARY_MATRIX(transpose)

}
