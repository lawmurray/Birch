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
 * @def INNER
 * 
 * Explicitly instantiate inner product.
 */
#define INNER(f) \
    INNER_SIG(f, real)
#define INNER_SIG(f, T) \
    template Array<T,1> f(const Array<T,2>&, const Array<T,1>&); \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&); \
    template Array<T,1> f(const Array<T,1>&, const Array<T,2>&, const Array<T,1>&); \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&, const Array<T,2>&);

namespace numbirch {
INNER(inner)
}
