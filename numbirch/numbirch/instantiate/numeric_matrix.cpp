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
 * @def MATRIX
 * 
 * Explicitly instantiate a unary function `f` over floating point matrices.
 * Use cases include transpose(), inv().
 */
#define MATRIX(f) \
    template Array<double,2> f(const Array<double,2>&); \
    template Array<float,2> f(const Array<float,2>&);

namespace numbirch {
MATRIX(cholinv)
MATRIX(inv)
MATRIX(transpose)
}
