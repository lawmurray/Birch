/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/reduce.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/reduce.hpp"
#endif

/**
 * @internal
 * 
 * @def REDUCE
 * 
 * Explicitly instantiate unary reduction.
 */
#define REDUCE(f) \
    REDUCE_FIRST(f, real) \
    REDUCE_FIRST(f, int) \
    REDUCE_FIRST(f, bool)
#define REDUCE_FIRST(f, R) \
    REDUCE_SECOND(f, R, real) \
    REDUCE_SECOND(f, R, int) \
    REDUCE_SECOND(f, R, bool)
#define REDUCE_SECOND(f, R, T) \
    REDUCE_SIG(f, R, ARRAY(T, 2)) \
    REDUCE_SIG(f, R, ARRAY(T, 1)) \
    REDUCE_SIG(f, R, ARRAY(T, 0))
#define REDUCE_SIG(f, R, T) \
    template Array<R,0> f<R>(const T&);

namespace numbirch {
REDUCE(sum)
REDUCE(count)
}
