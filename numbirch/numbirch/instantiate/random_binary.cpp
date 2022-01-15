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
#include "numbirch/common/binary.hpp"
#include "numbirch/common/random.hpp"

#define RANDOM_BINARY_REAL(f) \
    RANDOM_BINARY_FIRST(f, real)
#define RANDOM_BINARY_INT(f) \
    RANDOM_BINARY_FIRST(f, int)
#define RANDOM_BINARY_BOOL(f) \
    RANDOM_BINARY_FIRST(f, bool)
#define RANDOM_BINARY_FIRST(f, R) \
    RANDOM_BINARY_SECOND(f, R, real) \
    RANDOM_BINARY_SECOND(f, R, int) \
    RANDOM_BINARY_SECOND(f, R, bool)
#define RANDOM_BINARY_SECOND(f, R, T) \
    RANDOM_BINARY_THIRD(f, R, T, real) \
    RANDOM_BINARY_THIRD(f, R, T, int) \
    RANDOM_BINARY_THIRD(f, R, T, bool)
#define RANDOM_BINARY_THIRD(f, R, T, U) \
    RANDOM_BINARY_SIG(f, R, ARRAY(T, 2), ARRAY(U, 2)) \
    RANDOM_BINARY_SIG(f, R, ARRAY(T, 1), ARRAY(U, 1)) \
    RANDOM_BINARY_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0)) \
    RANDOM_BINARY_SIG(f, R, ARRAY(T, 0), U) \
    RANDOM_BINARY_SIG(f, R, T, ARRAY(U, 0)) \
    RANDOM_BINARY_SIG(f, R, T, U)
#define RANDOM_BINARY_SIG(f, R, T, U) \
    template explicit_t<R,T,U> f<T,U,int>(const T&, const U&);

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
