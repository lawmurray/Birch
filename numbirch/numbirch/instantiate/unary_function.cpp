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
#include "numbirch/common/random.hpp"

#define UNARY_ARITHMETIC(f) \
    UNARY_FIRST(f, real) \
    UNARY_FIRST(f, int) \
    UNARY_FIRST(f, bool)
#define UNARY_FIRST(f, R) \
    UNARY_SECOND(f, R, real) \
    UNARY_SECOND(f, R, int) \
    UNARY_SECOND(f, R, bool)
#define UNARY_SECOND(f, R, T) \
    UNARY_SIG(f, R, ARRAY(T, 2)) \
    UNARY_SIG(f, R, ARRAY(T, 1)) \
    UNARY_SIG(f, R, ARRAY(T, 0)) \
    UNARY_SIG(f, R, T)
#define UNARY_SIG(f, R, T) \
    template explicit_t<R,T> f<R,T,int>(const T&);

#define UNARY_FLOATING_POINT(f) \
    UNARY_FIRST(f, real)

namespace numbirch {
UNARY_ARITHMETIC(abs)
UNARY_FLOATING_POINT(acos)
UNARY_FLOATING_POINT(asin)
UNARY_FLOATING_POINT(atan)
UNARY_ARITHMETIC(cast)
UNARY_ARITHMETIC(ceil)
UNARY_FLOATING_POINT(cos)
UNARY_FLOATING_POINT(cosh)
UNARY_FLOATING_POINT(digamma)
UNARY_FLOATING_POINT(exp)
UNARY_FLOATING_POINT(expm1)
UNARY_ARITHMETIC(floor)
UNARY_FLOATING_POINT(lfact)
UNARY_FLOATING_POINT(lgamma)
UNARY_FLOATING_POINT(log)
UNARY_FLOATING_POINT(log1p)
UNARY_ARITHMETIC(rectify)
UNARY_ARITHMETIC(round)
UNARY_FLOATING_POINT(sin)
UNARY_FLOATING_POINT(sinh)
UNARY_FLOATING_POINT(sqrt)
UNARY_FLOATING_POINT(tan)
UNARY_FLOATING_POINT(tanh)

UNARY_ARITHMETIC(simulate_bernoulli)
UNARY_FLOATING_POINT(simulate_chi_squared)
UNARY_FLOATING_POINT(simulate_exponential)
UNARY_ARITHMETIC(simulate_poisson)
UNARY_FLOATING_POINT(simulate_student_t)

}
