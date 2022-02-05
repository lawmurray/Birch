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

#define REDUCE_SUM_GRAD(f) \
    REDUCE_SUM_GRAD_FIRST(f, real) \
    REDUCE_SUM_GRAD_FIRST(f, int) \
    REDUCE_SUM_GRAD_FIRST(f, bool)
#define REDUCE_SUM_GRAD_FIRST(f, T) \
    REDUCE_SUM_GRAD_SIG(f, ARRAY(T, 2)) \
    REDUCE_SUM_GRAD_SIG(f, ARRAY(T, 1)) \
    REDUCE_SUM_GRAD_SIG(f, ARRAY(T, 0)) \
    REDUCE_SUM_GRAD_SIG(f, T)
#define REDUCE_SUM_GRAD_SIG(f, T) \
    template default_t<T> f(const Array<real,0>&, \
        const Array<value_t<T>,0>&, const T&);

namespace numbirch {
REDUCE_SUM_GRAD(sum_grad)
}