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
 * @def OUTER
 * 
 * Explicitly instantiate outer product of vectors.
 */
#define OUTER(f) \
    OUTER_SIG(f, double) \
    OUTER_SIG(f, float)
#define OUTER_SIG(f, T) \
    template Array<T,2> f(const Array<T,1>&, const Array<T,1>&);

namespace numbirch {
OUTER(outer)
}
