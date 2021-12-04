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
 * @def DIAGONAL
 * 
 * Explicitly instantiate diagonal().
 */
#define DIAGONAL(f) \
    DIAGONAL_FIRST(f, double) \
    DIAGONAL_FIRST(f, float) \
    DIAGONAL_FIRST(f, int) \
    DIAGONAL_FIRST(f, bool)
#define DIAGONAL_FIRST(f, R) \
    DIAGONAL_DIM(f, R, double) \
    DIAGONAL_DIM(f, R, float) \
    DIAGONAL_DIM(f, R, int) \
    DIAGONAL_DIM(f, R, bool)
#define DIAGONAL_DIM(f, R, T) \
    DIAGONAL_SIG(f, R, T) \
    DIAGONAL_SIG(f, R, ARRAY(T, 0))
#define DIAGONAL_SIG(f, R, T) \
    template Array<R,2> f<R,T,int>(const T& x, const int n);

namespace numbirch {
DIAGONAL(diagonal)
}
