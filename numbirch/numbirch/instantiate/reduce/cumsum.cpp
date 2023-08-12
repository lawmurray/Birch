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

#define CUMSUM(f) \
    CUMSUM_FIRST(f, real) \
    CUMSUM_FIRST(f, int) \
    CUMSUM_FIRST(f, bool)
#define CUMSUM_FIRST(f, T) \
    CUMSUM_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    CUMSUM_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    CUMSUM_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    CUMSUM_SIG(f, T)
#define CUMSUM_SIG(f, T) \
    template T f(const T&);

namespace numbirch {
CUMSUM(cumsum)
}
