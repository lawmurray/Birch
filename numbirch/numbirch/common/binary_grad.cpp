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
#include "numbirch/common/binary.hpp"

/**
 * @internal
 * 
 * @def BINARY_GRAD
 * 
 * Explicitly instantiate the gradient of a binary transformation `f`.
 */
#define BINARY_GRAD(f) \
    BINARY_GRAD_FIRST(f, double) \
    BINARY_GRAD_FIRST(f, float)
#define BINARY_GRAD_FIRST(f, G) \
    BINARY_GRAD_SECOND(f, G, double) \
    BINARY_GRAD_SECOND(f, G, float) \
    BINARY_GRAD_SECOND(f, G, int) \
    BINARY_GRAD_SECOND(f, G, bool)
#define BINARY_GRAD_SECOND(f, G, T) \
    BINARY_GRAD_DIM(f, G, T, double) \
    BINARY_GRAD_DIM(f, G, T, float) \
    BINARY_GRAD_DIM(f, G, T, int) \
    BINARY_GRAD_DIM(f, G, T, bool)
#define BINARY_GRAD_DIM(f, G, T, U) \
    BINARY_GRAD_SIG(f, ARRAY(G, 2), ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, ARRAY(G, 1), ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, ARRAY(G, 0), T, ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, G, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, G, ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, G, T, ARRAY(U, 0))
#define BINARY_GRAD_SIG(f, G, T, U) \
    template std::pair<implicit_t<G,T,U>,implicit_t<G,T,U>> f<G,T,U>(const G&, \
        const T&, const U&);

namespace numbirch {
BINARY_GRAD(lchoose_grad)
}
