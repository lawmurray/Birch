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
 * @def SINGLE_VECTOR
 * 
 * For single().
 */
#define SINGLE_VECTOR(f) \
    SINGLE_VECTOR_FIRST(f, double) \
    SINGLE_VECTOR_FIRST(f, float) \
    SINGLE_VECTOR_FIRST(f, int) \
    SINGLE_VECTOR_FIRST(f, bool)
#define SINGLE_VECTOR_FIRST(f, R) \
    SINGLE_VECTOR_SIG(f, R, int) \
    SINGLE_VECTOR_SIG(f, R, ARRAY(int, 0))
#define SINGLE_VECTOR_SIG(f, R, T) \
    template Array<R,1> f<R>(const T& i, const int n);

namespace numbirch {
SINGLE_VECTOR(single)
}
