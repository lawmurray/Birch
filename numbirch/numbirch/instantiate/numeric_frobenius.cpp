/**
 * @file
 */
#include "numbirch/numeric.hpp"
#include "numbirch/array.hpp"
#include "numbirch/reduce.hpp"

#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.hpp"
#endif

/**
 * @internal
 * 
 * @def FROBENIUS
 * 
 * Explicitly instantiate frobenius().
 */
#define FROBENIUS(f) \
    FROBENIUS_SIG(f, double) \
    FROBENIUS_SIG(f, float)
#define FROBENIUS_SIG(f, T) \
    template Array<T,0> f(const Array<T,2>&, const Array<T,2>&);

namespace numbirch {
FROBENIUS(frobenius)
}
