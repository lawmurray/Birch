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

#define REDUCE_SUM(f) \
    REDUCE_SUM_FIRST(f, real) \
    REDUCE_SUM_FIRST(f, int) \
    REDUCE_SUM_FIRST(f, bool)
#define REDUCE_SUM_FIRST(f, T) \
    REDUCE_SUM_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    REDUCE_SUM_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    REDUCE_SUM_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    REDUCE_SUM_SIG(f, T)
#define REDUCE_SUM_SIG(f, T) \
    template Array<value_t<T>,0> f(const T&);

namespace numbirch {
REDUCE_SUM(sum)
REDUCE_SUM(min)
REDUCE_SUM(max)
}
