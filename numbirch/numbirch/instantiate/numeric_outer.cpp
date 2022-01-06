/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.hpp"
#endif

/**
 * @internal
 * 
 * @def OUTER
 * 
 * Explicitly instantiate outer product of vectors.
 */
#define OUTER(f) \
    OUTER_SIG(f, real)
#define OUTER_SIG(f, T) \
    template Array<T,2> f(const Array<T,1>&, const Array<T,1>&);

namespace numbirch {
OUTER(outer)
}
