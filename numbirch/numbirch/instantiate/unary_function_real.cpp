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
#include "numbirch/common/unary.hpp"

#define UNARY_FUNCTION_REAL(f) \
    UNARY_FUNCTION_REAL_FIRST(f, real) \
    UNARY_FUNCTION_REAL_FIRST(f, int) \
    UNARY_FUNCTION_REAL_FIRST(f, bool)
#define UNARY_FUNCTION_REAL_FIRST(f, T) \
    UNARY_FUNCTION_REAL_SIG(f, ARRAY(T, 2)) \
    UNARY_FUNCTION_REAL_SIG(f, ARRAY(T, 1)) \
    UNARY_FUNCTION_REAL_SIG(f, ARRAY(T, 0)) \
    UNARY_FUNCTION_REAL_SIG(f, T)
#define UNARY_FUNCTION_REAL_SIG(f, T) \
    template default_t<T> f<T,int>(const T&);

namespace numbirch {
UNARY_FUNCTION_REAL(acos)
UNARY_FUNCTION_REAL(asin)
UNARY_FUNCTION_REAL(atan)
UNARY_FUNCTION_REAL(cos)
UNARY_FUNCTION_REAL(cosh)
UNARY_FUNCTION_REAL(digamma)
UNARY_FUNCTION_REAL(exp)
UNARY_FUNCTION_REAL(expm1)
UNARY_FUNCTION_REAL(lfact)
UNARY_FUNCTION_REAL(lgamma)
UNARY_FUNCTION_REAL(log)
UNARY_FUNCTION_REAL(log1p)
UNARY_FUNCTION_REAL(sin)
UNARY_FUNCTION_REAL(sinh)
UNARY_FUNCTION_REAL(sqrt)
UNARY_FUNCTION_REAL(tan)
UNARY_FUNCTION_REAL(tanh)
}
