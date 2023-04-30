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

#define REDUCE_CUMSUM(f) \
    REDUCE_CUMSUM_FIRST(f, real) \
    REDUCE_CUMSUM_FIRST(f, int) \
    REDUCE_CUMSUM_FIRST(f, bool)
#define REDUCE_CUMSUM_FIRST(f, T) \
    REDUCE_CUMSUM_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    REDUCE_CUMSUM_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    REDUCE_CUMSUM_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    REDUCE_CUMSUM_SIG(f, T)
#define REDUCE_CUMSUM_SIG(f, T) \
    template T f(const T&);

namespace numbirch {
REDUCE_CUMSUM(cumsum)
}
