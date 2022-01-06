/**
 * @file
 */
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
 * @def STANDARD_MATRIX
 */
#define STANDARD_MATRIX(f) \
    STANDARD_MATRIX_SIG(f, real)
#define STANDARD_MATRIX_SIG(f, R) \
    template Array<R,2> f<R>(const int m, const int n);

namespace numbirch {
STANDARD_MATRIX(standard_gaussian)
}
