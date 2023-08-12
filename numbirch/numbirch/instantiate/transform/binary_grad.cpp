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

#define BINARY_GRAD(f) \
    BINARY_GRAD_FIRST(f, real) \
    BINARY_GRAD_FIRST(f, int) \
    BINARY_GRAD_FIRST(f, bool)
#define BINARY_GRAD_FIRST(f, T) \
    BINARY_GRAD_SECOND(f, T, real) \
    BINARY_GRAD_SECOND(f, T, int) \
    BINARY_GRAD_SECOND(f, T, bool)
#define BINARY_GRAD_SECOND(f, T, U) \
    BINARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 1)) \
    BINARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, T, NUMBIRCH_ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, T, U) \
    BINARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 2), U) \
    BINARY_GRAD_SIG(f, T, NUMBIRCH_ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 1)) \
    BINARY_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 1), U) \
    BINARY_GRAD_SIG(f, T, NUMBIRCH_ARRAY(U, 1))
#define BINARY_GRAD_SIG(f, T, U) \
    template real_t<T> f ## 1<T,U>(const real_t<T,U>&, const T&, const U&); \
    template real_t<U> f ## 2<T,U>(const real_t<T,U>&, const T&, const U&);

namespace numbirch {
BINARY_GRAD(equal_grad)
BINARY_GRAD(not_equal_grad)
BINARY_GRAD(less_grad)
BINARY_GRAD(less_or_equal_grad)
BINARY_GRAD(greater_grad)
BINARY_GRAD(greater_or_equal_grad)
BINARY_GRAD(add_grad)
BINARY_GRAD(copysign_grad)
BINARY_GRAD(div_grad)
BINARY_GRAD(hadamard_grad)
BINARY_GRAD(lbeta_grad)
BINARY_GRAD(lchoose_grad)
BINARY_GRAD(lgamma_grad)
BINARY_GRAD(logical_and_grad)
BINARY_GRAD(logical_or_grad)
BINARY_GRAD(pow_grad)
BINARY_GRAD(sub_grad)
}
