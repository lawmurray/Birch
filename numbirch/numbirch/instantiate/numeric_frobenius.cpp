/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.hpp"
#endif

#define FROBENIUS(f) \
    FROBENIUS_SIG(f, real)
#define FROBENIUS_SIG(f, T) \
    template Array<T,0> f(const Array<T,2>&); \
    template Array<T,0> f(const Array<T,2>&, const Array<T,2>&);

namespace numbirch {
FROBENIUS(frobenius)
}
