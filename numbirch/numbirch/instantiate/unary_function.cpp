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

#define UNARY_FUNCTION(f) \
    UNARY_FUNCTION_FIRST(f, real) \
    UNARY_FUNCTION_FIRST(f, int) \
    UNARY_FUNCTION_FIRST(f, bool)
#define UNARY_FUNCTION_FIRST(f, T) \
    UNARY_FUNCTION_SIG(f, ARRAY(T, 2)) \
    UNARY_FUNCTION_SIG(f, ARRAY(T, 1)) \
    UNARY_FUNCTION_SIG(f, ARRAY(T, 0)) \
    UNARY_FUNCTION_SIG(f, T)
#define UNARY_FUNCTION_SIG(f, T) \
    template T f<T,int>(const T&);

namespace numbirch {
UNARY_FUNCTION(abs)
UNARY_FUNCTION(ceil)
UNARY_FUNCTION(floor)
UNARY_FUNCTION(rectify)
UNARY_FUNCTION(round)
}
