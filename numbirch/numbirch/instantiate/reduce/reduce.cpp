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

#define REDUCE(f) \
    REDUCE_FIRST(f, real) \
    REDUCE_FIRST(f, int) \
    REDUCE_FIRST(f, bool)
#define REDUCE_FIRST(f, T) \
    REDUCE_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    REDUCE_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    REDUCE_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    REDUCE_SIG(f, T)
#define REDUCE_SIG(f, T) \
    template Array<value_t<T>,0> f(const T&);

namespace numbirch {
REDUCE(sum)
REDUCE(min)
REDUCE(max)
}
