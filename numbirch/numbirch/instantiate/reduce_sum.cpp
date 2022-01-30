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

#define REDUCE_SUM(f) \
    REDUCE_SUM_FIRST(f, real) \
    REDUCE_SUM_FIRST(f, int) \
    REDUCE_SUM_FIRST(f, bool)
#define REDUCE_SUM_FIRST(f, T) \
    REDUCE_SUM_SIG(f, ARRAY(T, 2)) \
    REDUCE_SUM_SIG(f, ARRAY(T, 1)) \
    REDUCE_SUM_SIG(f, ARRAY(T, 0))
#define REDUCE_SUM_SIG(f, T) \
    template Array<value_t<T>,0> f(const T&);

namespace numbirch {
REDUCE_SUM(sum)
}
