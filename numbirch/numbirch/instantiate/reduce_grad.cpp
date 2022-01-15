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

#define REDUCE_GRAD(f) \
    REDUCE_GRAD_FIRST(f, real) \
    REDUCE_GRAD_FIRST(f, int) \
    REDUCE_GRAD_FIRST(f, bool)
#define REDUCE_GRAD_FIRST(f, R) \
    REDUCE_GRAD_SECOND(f, R, real) \
    REDUCE_GRAD_SECOND(f, R, int) \
    REDUCE_GRAD_SECOND(f, R, bool)
#define REDUCE_GRAD_SECOND(f, R, T) \
    REDUCE_GRAD_SIG(f, ARRAY(R, 0), ARRAY(T, 2)) \
    REDUCE_GRAD_SIG(f, ARRAY(R, 0), ARRAY(T, 1)) \
    REDUCE_GRAD_SIG(f, ARRAY(R, 0), ARRAY(T, 0)) \
    REDUCE_GRAD_SIG(f, ARRAY(R, 0), T) \
    REDUCE_GRAD_SIG(f, R, ARRAY(T, 2)) \
    REDUCE_GRAD_SIG(f, R, ARRAY(T, 1)) \
    REDUCE_GRAD_SIG(f, R, ARRAY(T, 0)) \
    REDUCE_GRAD_SIG(f, R, T)
#define REDUCE_GRAD_SIG(f, R, T) \
    template default_t<T> f<R,T,int>(const Array<real,0>&, \
        const Array<R,0>&, const T&);

namespace numbirch {
REDUCE_GRAD(sum_grad)
REDUCE_GRAD(count_grad)
}
