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

#define BINARY_GRAD(f, R) \
    BINARY_GRAD_FIRST(f, R, real) \
    BINARY_GRAD_FIRST(f, R, int) \
    BINARY_GRAD_FIRST(f, R, bool)
#define BINARY_GRAD_FIRST(f, R, T) \
    BINARY_GRAD_SECOND(f, R, T, real) \
    BINARY_GRAD_SECOND(f, R, T, int) \
    BINARY_GRAD_SECOND(f, R, T, bool)
#define BINARY_GRAD_SECOND(f, R, T, U) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, R, T, ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, R, T, U) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 2), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 0), ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 2), U) \
    BINARY_GRAD_SIG(f, R, T, ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 1), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 0), ARRAY(U, 1)) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 1), U) \
    BINARY_GRAD_SIG(f, R, T, ARRAY(U, 1))
#define BINARY_GRAD_SIG(f, R, T, U) \
    template std::pair<real_t<T>,real_t<U>> f<T,U,int>( \
        const real_t<T,U>&, const R<T,U>&, const T&, const U&);

#define BINARY_ARITHMETIC_GRAD(f) BINARY_GRAD(f, implicit_t)
#define BINARY_REAL_GRAD(f) BINARY_GRAD(f, real_t)
#define BINARY_BOOL_GRAD(f) BINARY_GRAD(f, bool_t)

namespace numbirch {
BINARY_BOOL_GRAD(and_grad)
BINARY_BOOL_GRAD(or_grad)
BINARY_BOOL_GRAD(equal_grad)
BINARY_BOOL_GRAD(not_equal_grad)
BINARY_BOOL_GRAD(less_grad)
BINARY_BOOL_GRAD(less_or_equal_grad)
BINARY_BOOL_GRAD(greater_grad)
BINARY_BOOL_GRAD(greater_or_equal_grad)
BINARY_ARITHMETIC_GRAD(copysign_grad)
BINARY_ARITHMETIC_GRAD(hadamard_grad)
BINARY_REAL_GRAD(lbeta_grad)
BINARY_REAL_GRAD(lchoose_grad)
BINARY_REAL_GRAD(lgamma_grad)
BINARY_REAL_GRAD(pow_grad)
}
