/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/array.hpp"

/**
 * @internal
 * 
 * @def DIAGONAL
 * 
 * Explicitly instantiate diagonal().
 */
#define DIAGONAL(f) \
    DIAGONAL_FIRST(f, real) \
    DIAGONAL_FIRST(f, int) \
    DIAGONAL_FIRST(f, bool)
#define DIAGONAL_FIRST(f, T) \
    DIAGONAL_SIG(f, T) \
    DIAGONAL_SIG(f, NUMBIRCH_ARRAY(T, 0))
#define DIAGONAL_SIG(f, T) \
    template Array<value_t<T>,2> f<T,int>(const T& x, const int n);

namespace numbirch {
DIAGONAL(diagonal)
}
