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

#define UNARY(f, R) \
    UNARY_FIRST(f, R, real) \
    UNARY_FIRST(f, R, int) \
    UNARY_FIRST(f, R, bool)
#define UNARY_FIRST(f, R, T) \
    UNARY_SIG(f, R, NUMBIRCH_ARRAY(T, 2)) \
    UNARY_SIG(f, R, NUMBIRCH_ARRAY(T, 1)) \
    UNARY_SIG(f, R, NUMBIRCH_ARRAY(T, 0)) \
    UNARY_SIG(f, R, T)
#define UNARY_SIG(f, R, T) \
    template R<T> f<T>(const T&);

#define UNARY_ARITHMETIC(f) UNARY(f, implicit_t)
#define UNARY_REAL(f) UNARY(f, real_t)
#define UNARY_BOOL(f) UNARY(f, bool_t)

namespace numbirch {
UNARY_ARITHMETIC(abs)
UNARY_REAL(acos)
UNARY_REAL(asin)
UNARY_REAL(atan)
UNARY_ARITHMETIC(ceil)
UNARY_REAL(cos)
UNARY_REAL(cosh)
UNARY_REAL(digamma)
UNARY_REAL(erf)
UNARY_REAL(exp)
UNARY_REAL(expm1)
UNARY_ARITHMETIC(floor)
UNARY_BOOL(isfinite)
UNARY_BOOL(isinf)
UNARY_BOOL(isnan)
UNARY_REAL(lfact)
UNARY_REAL(lgamma)
UNARY_REAL(log)
UNARY_REAL(log1p)
UNARY_BOOL(logical_not)
UNARY_ARITHMETIC(neg)
UNARY_ARITHMETIC(pos)
UNARY_ARITHMETIC(rectify)
UNARY_ARITHMETIC(round)
UNARY_REAL(sin)
UNARY_REAL(sinh)
UNARY_REAL(sqrt)
UNARY_REAL(tan)
UNARY_REAL(tanh)
}
