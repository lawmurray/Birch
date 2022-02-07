/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#include "numbirch/cuda/random.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#include "numbirch/eigen/random.hpp"
#endif
#include "numbirch/common/transform.hpp"

#define BINARY(f, R) \
    BINARY_FIRST(f, R, real) \
    BINARY_FIRST(f, R, int) \
    BINARY_FIRST(f, R, bool)
#define BINARY_FIRST(f, R, T) \
    BINARY_SECOND(f, R, T, real) \
    BINARY_SECOND(f, R, T, int) \
    BINARY_SECOND(f, R, T, bool)
#define BINARY_SECOND(f, R, T, U) \
    BINARY_SIG(f, R, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_SIG(f, R, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_SIG(f, R, ARRAY(T, 0), U) \
    BINARY_SIG(f, R, T, ARRAY(U, 0)) \
    BINARY_SIG(f, R, T, U) \
    BINARY_SIG(f, R, ARRAY(T, 2), ARRAY(U, 0)) \
    BINARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 2)) \
    BINARY_SIG(f, R, ARRAY(T, 2), U) \
    BINARY_SIG(f, R, T, ARRAY(U, 2)) \
    BINARY_SIG(f, R, ARRAY(T, 1), ARRAY(U, 0)) \
    BINARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 1)) \
    BINARY_SIG(f, R, ARRAY(T, 1), U) \
    BINARY_SIG(f, R, T, ARRAY(U, 1))
#define BINARY_SIG(f, R, T, U) \
    template R<T,U> f<T,U,int>(const T&, const U&);

#define BINARY_ARITHMETIC(f) BINARY(f, implicit_t)
#define BINARY_REAL(f) BINARY(f, real_t)
#define BINARY_BOOL(f) BINARY(f, bool_t)

namespace numbirch {
BINARY_ARITHMETIC(copysign)
BINARY_REAL(digamma)
BINARY_REAL(gamma_p)
BINARY_REAL(gamma_q)
BINARY_ARITHMETIC(hadamard)
BINARY_REAL(lbeta)
BINARY_REAL(lchoose)
BINARY_REAL(lgamma)
BINARY_REAL(pow)
}
