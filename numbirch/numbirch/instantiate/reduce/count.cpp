/**
 * @file
 */
#include "numbirch/common/reduce.inl"
#ifdef BACKEND_CUDA
#include "numbirch/cuda/reduce.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/reduce.inl"
#endif

#define REDUCE_COUNT(f) \
    REDUCE_COUNT_FIRST(f, real) \
    REDUCE_COUNT_FIRST(f, int) \
    REDUCE_COUNT_FIRST(f, bool)
#define REDUCE_COUNT_FIRST(f, T) \
    REDUCE_COUNT_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    REDUCE_COUNT_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    REDUCE_COUNT_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    REDUCE_COUNT_SIG(f, T)
#define REDUCE_COUNT_SIG(f, T) \
    template Array<int,0> f(const T&);

namespace numbirch {
REDUCE_COUNT(count)
}
