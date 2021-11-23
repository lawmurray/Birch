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
 * @def SUM
 * 
 * Explicitly instantiate sum().
 */
#define SUM(f) \
    SUM_FIRST(f, double) \
    SUM_FIRST(f, float) \
    SUM_FIRST(f, int) \
    SUM_FIRST(f, bool)
#define SUM_FIRST(f, R) \
    SUM_DIM(f, R, double) \
    SUM_DIM(f, R, float) \
    SUM_DIM(f, R, int) \
    SUM_DIM(f, R, bool)
#define SUM_DIM(f, R, T) \
    SUM_SIG(f, R, ARRAY(T, 2)) \
    SUM_SIG(f, R, ARRAY(T, 1)) \
    SUM_SIG(f, R, ARRAY(T, 0))
#define SUM_SIG(f, R, T) \
    template Array<R,0> f<R>(const T&);

namespace numbirch {
SUM(sum)
}
