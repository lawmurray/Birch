/**
 * @file
 */
#include "numbirch/numeric.hpp"
#include "numbirch/array.hpp"
#include "numbirch/reduce.hpp"

#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.hpp"
#endif

/**
 * @internal
 * 
 * @def COUNT
 * 
 * Explicitly instantiate count().
 */
#define COUNT(f) \
    COUNT_FIRST(f, double) \
    COUNT_FIRST(f, float) \
    COUNT_FIRST(f, int) \
    COUNT_FIRST(f, bool)
#define COUNT_FIRST(f, R) \
    COUNT_DIM(f, R, double) \
    COUNT_DIM(f, R, float) \
    COUNT_DIM(f, R, int) \
    COUNT_DIM(f, R, bool)
#define COUNT_DIM(f, R, T) \
    COUNT_SIG(f, R, ARRAY(T, 2)) \
    COUNT_SIG(f, R, ARRAY(T, 1)) \
    COUNT_SIG(f, R, ARRAY(T, 0))
#define COUNT_SIG(f, R, T) \
    template Array<R,0> f<R>(const T&);

namespace numbirch {
COUNT(count)
}
