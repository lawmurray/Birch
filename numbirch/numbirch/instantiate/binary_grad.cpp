/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/binary.hpp"

#define BINARY_GRAD(f) \
    BINARY_GRAD_FIRST(f, real)
#define BINARY_GRAD_FIRST(f, G) \
    BINARY_GRAD_SECOND(f, G, real) \
    BINARY_GRAD_SECOND(f, G, int) \
    BINARY_GRAD_SECOND(f, G, bool)
#define BINARY_GRAD_SECOND(f, G, T) \
    BINARY_GRAD_THIRD(f, G, T, real) \
    BINARY_GRAD_THIRD(f, G, T, int) \
    BINARY_GRAD_THIRD(f, G, T, bool)
#define BINARY_GRAD_THIRD(f, G, T, U) \
    BINARY_GRAD_SIG(f, ARRAY(G, 2), ARRAY(T, 2), ARRAY(U, 2)) \
    BINARY_GRAD_SIG(f, ARRAY(G, 1), ARRAY(T, 1), ARRAY(U, 1)) \
    BINARY_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, ARRAY(G, 0), ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, ARRAY(G, 0), T, ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, G, ARRAY(T, 0), ARRAY(U, 0)) \
    BINARY_GRAD_SIG(f, G, ARRAY(T, 0), U) \
    BINARY_GRAD_SIG(f, G, T, ARRAY(U, 0))
#define BINARY_GRAD_SIG(f, G, T, U) \
    template std::pair<default_t<G,T,U>,default_t<G,T,U>> f<G,T,U>(const G&, \
        const T&, const U&);

namespace numbirch {
BINARY_GRAD(and_grad)
BINARY_GRAD(or_grad)
BINARY_GRAD(equal_grad)
BINARY_GRAD(not_equal_grad)
BINARY_GRAD(less_grad)
BINARY_GRAD(less_or_equal_grad)
BINARY_GRAD(greater_grad)
BINARY_GRAD(greater_or_equal_grad)

BINARY_GRAD(copysign_grad)
BINARY_GRAD(hadamard_grad)
BINARY_GRAD(lbeta_grad)
BINARY_GRAD(lchoose_grad)
BINARY_GRAD(lgamma_grad)
BINARY_GRAD(pow_grad)

}
