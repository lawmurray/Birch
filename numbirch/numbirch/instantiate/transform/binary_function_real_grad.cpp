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

#define BINARY_FUNCTION_REAL_GRAD(f) \
    BINARY_FUNCTION_REAL_GRAD_FIRST(f, real) \
    BINARY_FUNCTION_REAL_GRAD_FIRST(f, int) \
    BINARY_FUNCTION_REAL_GRAD_FIRST(f, bool)
#define BINARY_FUNCTION_REAL_GRAD_FIRST(f, T) \
    BINARY_FUNCTION_REAL_GRAD_SECOND(f, T, real) \
    BINARY_FUNCTION_REAL_GRAD_SECOND(f, T, int) \
    BINARY_FUNCTION_REAL_GRAD_SECOND(f, T, bool)
#define BINARY_FUNCTION_REAL_GRAD_SECOND(f, T, U) \
    BINARY_FUNCTION_REAL_GRAD_SIG(f, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_FUNCTION_REAL_GRAD_SIG(f, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_FUNCTION_REAL_GRAD_SIG(f, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_FUNCTION_REAL_GRAD_SIG(f, ARRAY(T, 0), U) \
    BINARY_FUNCTION_REAL_GRAD_SIG(f, T, ARRAY(U, 0)) \
    BINARY_FUNCTION_REAL_GRAD_SIG(f, T, U)
#define BINARY_FUNCTION_REAL_GRAD_SIG(f, T, U) \
    template std::pair<default_t<T>,default_t<U>> f<T,U,int>( \
        const default_t<T,U>&, const default_t<T,U>&, const T&, const U&);

namespace numbirch {
BINARY_FUNCTION_REAL_GRAD(lbeta_grad)
BINARY_FUNCTION_REAL_GRAD(lchoose_grad)
BINARY_FUNCTION_REAL_GRAD(lgamma_grad)
BINARY_FUNCTION_REAL_GRAD(pow_grad)
}
