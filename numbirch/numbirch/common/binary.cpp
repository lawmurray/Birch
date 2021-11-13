/**
 * @file
 */
#include "numbirch/numeric.hpp"

#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/binary.hpp"

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

namespace numbirch {
BINARY_ARITHMETIC(operator+)
BINARY_ARITHMETIC(operator-)
BINARY_SCALAR_MULTIPLY(operator*)
BINARY_SCALAR_DIVIDE(operator/)
BINARY_ARITHMETIC(operator&&)
BINARY_ARITHMETIC(operator||)
BINARY_ARITHMETIC(operator==)
BINARY_ARITHMETIC(operator!=)
BINARY_ARITHMETIC(operator<)
BINARY_ARITHMETIC(operator<=)
BINARY_ARITHMETIC(operator>)
BINARY_ARITHMETIC(operator>=)
BINARY_ARITHMETIC(copysign)
BINARY_FLOATING_POINT(digamma)
BINARY_FLOATING_POINT(gamma_p)
BINARY_FLOATING_POINT(gamma_q)
BINARY_ARITHMETIC(hadamard)
BINARY_FLOATING_POINT(lbeta)
BINARY_FLOATING_POINT(lchoose)
BINARY_GRAD(lchoose_grad)
BINARY_FLOATING_POINT(lgamma)
BINARY_FLOATING_POINT(pow)

}
