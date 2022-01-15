/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/binary.hpp"

#define BINARY_GRAD(f) \
    BINARY_GRAD_FIRST(f, real) \
    BINARY_GRAD_FIRST(f, int) \
    BINARY_GRAD_FIRST(f, bool)
#define BINARY_GRAD_FIRST(f, R) \
    BINARY_GRAD_SECOND(f, R, real) \
    BINARY_GRAD_SECOND(f, R, int) \
    BINARY_GRAD_SECOND(f, R, bool)
#define BINARY_GRAD_SECOND(f, R, T) \
    BINARY_GRAD_THIRD(f, R, T, real) \
    BINARY_GRAD_THIRD(f, R, T, int) \
    BINARY_GRAD_THIRD(f, R, T, bool)
#define BINARY_GRAD_THIRD(f, R, T, U) \
    BINARY_GRAD_SIG(f, ARRAY(R, 2), ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, ARRAY(R, 1), ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_GRAD_SIG(f, ARRAY(R, 0), ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, ARRAY(R, 0), ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, ARRAY(R, 0), T, ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, R, ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, R, T, ARRAY(U, 0))
#define BINARY_GRAD_SIG(f, R, T, U) \
    template std::pair<default_t<T>,default_t<U>> f<R,T,U,int>( \
        const default_t<T,U>&, const explicit_t<R,T,U>&, const T&, const U&);

#define BINARY_GRAD_FLOATING_POINT(f) \
    BINARY_GRAD_FIRST(f, real)

namespace numbirch {
BINARY_GRAD(and_grad)
BINARY_GRAD(or_grad)
BINARY_GRAD(equal_grad)
BINARY_GRAD(not_equal_grad)
BINARY_GRAD(less_grad)
BINARY_GRAD(less_or_equal_grad)
BINARY_GRAD(greater_grad)
BINARY_GRAD(greater_or_equal_grad)
BINARY_GRAD(copysign_grad)
BINARY_GRAD(hadamard_grad)
BINARY_GRAD_FLOATING_POINT(lbeta_grad)
BINARY_GRAD_FLOATING_POINT(lchoose_grad)
BINARY_GRAD_FLOATING_POINT(lgamma_grad)
BINARY_GRAD_FLOATING_POINT(pow_grad)

}
