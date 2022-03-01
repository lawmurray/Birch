/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/reduce.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/reduce.hpp"
#endif
#include "numbirch/common/reduce.hpp"

#define REDUCE_COUNT(f) \
    REDUCE_COUNT_FIRST(f, real) \
    REDUCE_COUNT_FIRST(f, int) \
    REDUCE_COUNT_FIRST(f, bool)
#define REDUCE_COUNT_FIRST(f, T) \
    REDUCE_COUNT_SIG(f, ARRAY(T, 2)) \
    REDUCE_COUNT_SIG(f, ARRAY(T, 1)) \
    REDUCE_COUNT_SIG(f, ARRAY(T, 0))
#define REDUCE_COUNT_SIG(f, T) \
    template Array<int,0> f(const T&);

namespace numbirch {
REDUCE_COUNT(count)
}