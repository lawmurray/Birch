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
 * @def SINGLE_MATRIX
 * 
 * For single().
 */
#define SINGLE_MATRIX(f) \
    SINGLE_MATRIX_FIRST(f, double) \
    SINGLE_MATRIX_FIRST(f, float) \
    SINGLE_MATRIX_FIRST(f, int) \
    SINGLE_MATRIX_FIRST(f, bool)
#define SINGLE_MATRIX_FIRST(f, R) \
    SINGLE_MATRIX_SIG(f, R, ARRAY(int, 0), ARRAY(int, 0)) \
    SINGLE_MATRIX_SIG(f, R, ARRAY(int, 0), int) \
    SINGLE_MATRIX_SIG(f, R, int, ARRAY(int, 0)) \
    SINGLE_MATRIX_SIG(f, R, int, int)
#define SINGLE_MATRIX_SIG(f, R, T, U) \
    template Array<R,2> f<R>(const T& i, const U& j, const int m, const int n);

namespace numbirch {
SINGLE_MATRIX(single)
}
