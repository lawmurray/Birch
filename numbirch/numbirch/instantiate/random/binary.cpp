/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#include "numbirch/cuda/random.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#include "numbirch/eigen/random.inl"
#endif
#include "numbirch/common/transform.inl"
#include "numbirch/common/random.inl"

#define RANDOM_BINARY(f, R) \
    RANDOM_BINARY_FIRST(f, R, real) \
    RANDOM_BINARY_FIRST(f, R, int) \
    RANDOM_BINARY_FIRST(f, R, bool)
#define RANDOM_BINARY_FIRST(f, R, T) \
    RANDOM_BINARY_SECOND(f, R, T, real) \
    RANDOM_BINARY_SECOND(f, R, T, int) \
    RANDOM_BINARY_SECOND(f, R, T, bool)
#define RANDOM_BINARY_SECOND(f, R, T, U) \
    RANDOM_BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 2)) \
    RANDOM_BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 1)) \
    RANDOM_BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0)) \
    RANDOM_BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), U) \
    RANDOM_BINARY_SIG(f, R, T, NUMBIRCH_ARRAY(U, 0)) \
    RANDOM_BINARY_SIG(f, R, T, U) \
    RANDOM_BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 0)) \
    RANDOM_BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 2)) \
    RANDOM_BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 2), U) \
    RANDOM_BINARY_SIG(f, R, T, NUMBIRCH_ARRAY(U, 2)) \
    RANDOM_BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 0)) \
    RANDOM_BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 1)) \
    RANDOM_BINARY_SIG(f, R, NUMBIRCH_ARRAY(T, 1), U) \
    RANDOM_BINARY_SIG(f, R, T, NUMBIRCH_ARRAY(U, 1))

#define RANDOM_BINARY_SIG(f, R, T, U) \
    template R<T,U> f<T,U>(const T&, const U&);

#define RANDOM_BINARY_REAL(f) RANDOM_BINARY(f, real_t)
#define RANDOM_BINARY_INT(f) RANDOM_BINARY(f, int_t)
#define RANDOM_BINARY_BOOL(f) RANDOM_BINARY(f, bool_t)

namespace numbirch {
RANDOM_BINARY_REAL(simulate_beta)
RANDOM_BINARY_INT(simulate_binomial)
RANDOM_BINARY_REAL(simulate_gamma)
RANDOM_BINARY_REAL(simulate_gaussian)
RANDOM_BINARY_INT(simulate_negative_binomial)
RANDOM_BINARY_REAL(simulate_weibull)
RANDOM_BINARY_REAL(simulate_uniform)
RANDOM_BINARY_INT(simulate_uniform_int)
}
