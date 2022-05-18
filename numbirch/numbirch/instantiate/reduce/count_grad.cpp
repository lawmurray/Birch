/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/reduce.hpp"

#define REDUCE_COUNT_GRAD(f) \
    REDUCE_COUNT_GRAD_FIRST(f, real) \
    REDUCE_COUNT_GRAD_FIRST(f, int) \
    REDUCE_COUNT_GRAD_FIRST(f, bool)
#define REDUCE_COUNT_GRAD_FIRST(f, T) \
    REDUCE_COUNT_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    REDUCE_COUNT_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    REDUCE_COUNT_GRAD_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    REDUCE_COUNT_GRAD_SIG(f, T)
#define REDUCE_COUNT_GRAD_SIG(f, T) \
    template real_t<T> f(const Array<real,0>&, const Array<int,0>&, \
        const T&);

namespace numbirch {
REDUCE_COUNT_GRAD(count_grad)
}
