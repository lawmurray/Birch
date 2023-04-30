/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#endif
#include "numbirch/common/transform.inl"

#define BINARY(f, R) \
    BINARY_FIRST(f, R, real) \
    BINARY_FIRST(f, R, int) \
    BINARY_FIRST(f, R, bool)
#define BINARY_FIRST(f, R, T) \
    BINARY_SECOND(f, R, T, real) \
    BINARY_SECOND(f, R, T, int) \
    BINARY_SECOND(f, R, T, bool)
#define BINARY_SECOND(f, R, T, U) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 2)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 1)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), U) \
    BINARY_SIG(f, R, T, NUMBIRCH_ARRAY(U, 0)) \
    BINARY_SIG(f, R, T, U) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 2)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 2), U) \
    BINARY_SIG(f, R, T, NUMBIRCH_ARRAY(U, 2)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 1)) \
    BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 1), U) \
    BINARY_SIG(f, R, T, NUMBIRCH_ARRAY(U, 1))
#define BINARY_SIG(f, R, T, U) \
    template R<T,U> f<T,U,int>(const T&, const U&);

#define BINARY_ARITHMETIC(f) BINARY(f, implicit_t)
#define BINARY_REAL(f) BINARY(f, real_t)
#define BINARY_BOOL(f) BINARY(f, bool_t)

namespace numbirch {
BINARY_ARITHMETIC(add)
BINARY_ARITHMETIC(copysign)
BINARY_ARITHMETIC(div)
BINARY_REAL(digamma)
BINARY_BOOL(equal)
BINARY_REAL(gamma_p)
BINARY_REAL(gamma_q)
BINARY_BOOL(greater)
BINARY_BOOL(greater_or_equal)
BINARY_ARITHMETIC(hadamard)
BINARY_REAL(lbeta)
BINARY_REAL(lchoose)
BINARY_BOOL(less)
BINARY_BOOL(less_or_equal)
BINARY_REAL(lgamma)
BINARY_BOOL(logical_and)
BINARY_BOOL(logical_or)
BINARY_BOOL(not_equal)
BINARY_REAL(pow)
BINARY_ARITHMETIC(sub)

}
