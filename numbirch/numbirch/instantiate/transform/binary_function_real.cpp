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
#include "numbirch/common/random.hpp"

#define BINARY_FUNCTION_REAL(f) \
    BINARY_FUNCTION_REAL_FIRST(f, real) \
    BINARY_FUNCTION_REAL_FIRST(f, int) \
    BINARY_FUNCTION_REAL_FIRST(f, bool)
#define BINARY_FUNCTION_REAL_FIRST(f, T) \
    BINARY_FUNCTION_REAL_SECOND(f, T, real) \
    BINARY_FUNCTION_REAL_SECOND(f, T, int) \
    BINARY_FUNCTION_REAL_SECOND(f, T, bool)
#define BINARY_FUNCTION_REAL_SECOND(f, T, U) \
    BINARY_FUNCTION_REAL_SIG(f, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_FUNCTION_REAL_SIG(f, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_FUNCTION_REAL_SIG(f, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_FUNCTION_REAL_SIG(f, ARRAY(T, 0), U) \
    BINARY_FUNCTION_REAL_SIG(f, T, ARRAY(U, 0)) \
    BINARY_FUNCTION_REAL_SIG(f, T, U)
#define BINARY_FUNCTION_REAL_SIG(f, T, U) \
    template default_t<T,U> f<T,U,int>(const T&, const U&);

namespace numbirch {
BINARY_FUNCTION_REAL(digamma)
BINARY_FUNCTION_REAL(gamma_p)
BINARY_FUNCTION_REAL(gamma_q)
BINARY_FUNCTION_REAL(lbeta)
BINARY_FUNCTION_REAL(lchoose)
BINARY_FUNCTION_REAL(lgamma)
BINARY_FUNCTION_REAL(pow)
}
