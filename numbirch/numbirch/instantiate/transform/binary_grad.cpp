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

#define BINARY_GRAD(f, R) \
    BINARY_GRAD_FIRST(f, R, real) \
    BINARY_GRAD_FIRST(f, R, int) \
    BINARY_GRAD_FIRST(f, R, bool)
#define BINARY_GRAD_FIRST(f, R, T) \
    BINARY_GRAD_SECOND(f, R, T, real) \
    BINARY_GRAD_SECOND(f, R, T, int) \
    BINARY_GRAD_SECOND(f, R, T, bool)
#define BINARY_GRAD_SECOND(f, R, T, U) \
    BINARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 1)) \
    BINARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, R, T, NUMBIRCH_ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, R, T, U) \
    BINARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 2), U) \
    BINARY_GRAD_SIG(f, R, T, NUMBIRCH_ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 1)) \
    BINARY_GRAD_SIG(f, R, NUMBIRCH_ARRAY(T, 1), U) \
    BINARY_GRAD_SIG(f, R, T, NUMBIRCH_ARRAY(U, 1))
#define BINARY_GRAD_SIG(f, R, T, U) \
    template real_t<T> f ## 1<T,U,int>(const real_t<T,U>&, const R<T,U>&, \
        const T&, const U&); \
    template real_t<U> f ## 2<T,U,int>(const real_t<T,U>&, const R<T,U>&, \
        const T&, const U&);

#define BINARY_ARITHMETIC_GRAD(f) BINARY_GRAD(f, implicit_t)
#define BINARY_REAL_GRAD(f) BINARY_GRAD(f, real_t)
#define BINARY_BOOL_GRAD(f) BINARY_GRAD(f, bool_t)

namespace numbirch {
BINARY_BOOL_GRAD(equal_grad)
BINARY_BOOL_GRAD(not_equal_grad)
BINARY_BOOL_GRAD(less_grad)
BINARY_BOOL_GRAD(less_or_equal_grad)
BINARY_BOOL_GRAD(greater_grad)
BINARY_BOOL_GRAD(greater_or_equal_grad)
BINARY_ARITHMETIC_GRAD(add_grad)
BINARY_ARITHMETIC_GRAD(copysign_grad)
BINARY_ARITHMETIC_GRAD(div_grad)
BINARY_ARITHMETIC_GRAD(hadamard_grad)
BINARY_REAL_GRAD(lbeta_grad)
BINARY_REAL_GRAD(lchoose_grad)
BINARY_REAL_GRAD(lgamma_grad)
BINARY_BOOL_GRAD(logical_and_grad)
BINARY_BOOL_GRAD(logical_or_grad)
BINARY_REAL_GRAD(pow_grad)
BINARY_ARITHMETIC_GRAD(sub_grad)
}
