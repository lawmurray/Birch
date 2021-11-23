/**
 * @file
 */
#include "numbirch/numeric.hpp"

#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/unary.hpp"

/**
 * @internal
 * 
 * @def UNARY_GRAD
 * 
 * Explicitly instantiate the gradient of a unary transformation `f`.
 */
#define UNARY_GRAD(f) \
    UNARY_GRAD_FIRST(f, double) \
    UNARY_GRAD_FIRST(f, float)
#define UNARY_GRAD_FIRST(f, G) \
    UNARY_GRAD_DIM(f, G, double) \
    UNARY_GRAD_DIM(f, G, float) \
    UNARY_GRAD_DIM(f, G, int) \
    UNARY_GRAD_DIM(f, G, bool)
#define UNARY_GRAD_DIM(f, G, T) \
    UNARY_GRAD_SIG(f, ARRAY(G, 2), ARRAY(T, 2)) \
    UNARY_GRAD_SIG(f, ARRAY(G, 1), ARRAY(T, 1)) \
    UNARY_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 0)) \
    UNARY_GRAD_SIG(f, ARRAY(G, 0), T) \
    UNARY_GRAD_SIG(f, G, ARRAY(T, 0)) \
    UNARY_GRAD_SIG(f, G, T)
#define UNARY_GRAD_SIG(f, G, T) \
    template implicit_t<G,T> f<G,T,int>(const G&, const T&);

namespace numbirch {
UNARY_GRAD(lfact_grad)
UNARY_GRAD(rectify_grad)
}
