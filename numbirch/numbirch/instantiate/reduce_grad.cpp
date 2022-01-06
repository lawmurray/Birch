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

/**
 * @internal
 * 
 * @def REDUCE_GRAD
 * 
 * Explicitly instantiate the gradient of a unary reduction `f`.
 */
#define REDUCE_GRAD(f) \
    REDUCE_GRAD_FIRST(f, real)
#define REDUCE_GRAD_FIRST(f, G) \
    REDUCE_GRAD_SECOND(f, G, real) \
    REDUCE_GRAD_SECOND(f, G, int) \
    REDUCE_GRAD_SECOND(f, G, bool)
#define REDUCE_GRAD_SECOND(f, G, T) \
    REDUCE_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 2)) \
    REDUCE_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 1)) \
    REDUCE_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 0)) \
    REDUCE_GRAD_SIG(f, ARRAY(G, 0), T) \
    REDUCE_GRAD_SIG(f, G, ARRAY(T, 2)) \
    REDUCE_GRAD_SIG(f, G, ARRAY(T, 1)) \
    REDUCE_GRAD_SIG(f, G, ARRAY(T, 0)) \
    REDUCE_GRAD_SIG(f, G, T)
#define REDUCE_GRAD_SIG(f, G, T) \
    template default_t<G,T> f<G,T,int>(const G&, const T&);

namespace numbirch {
REDUCE_GRAD(sum_grad)
REDUCE_GRAD(count_grad)
}
