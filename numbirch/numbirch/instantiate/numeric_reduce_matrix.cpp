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
 * @def REDUCE_MATRIX
 * 
 * Explicitly instantiate a unary reduction function `f` for matrices of
 * floating point types only. Use cases include ldet(), trace().
 */
#define REDUCE_MATRIX(f) \
    template Array<double,0> f(const Array<double,2>&); \
    template Array<float,0> f(const Array<float,2>&);

namespace numbirch {
REDUCE_MATRIX(lcholdet)
REDUCE_MATRIX(ldet)
REDUCE_MATRIX(trace)
}
