/**
 * @file
 */
#include "numbirch/random.hpp"

#ifdef BACKEND_CUDA
#include "numbirch/cuda/random.hpp"
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/random.hpp"
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/random.hpp"

/**
 * @internal
 * 
 * @def STANDARD_VECTOR
 */
#define STANDARD_VECTOR(f) \
    STANDARD_VECTOR_SIG(f, double) \
    STANDARD_VECTOR_SIG(f, float)
#define STANDARD_VECTOR_SIG(f, R) \
    template Array<R,1> f<R>(const int n);

namespace numbirch {
STANDARD_VECTOR(standard_gaussian)
}
