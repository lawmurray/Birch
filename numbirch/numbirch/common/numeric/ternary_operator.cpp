/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/numeric/ternary_operator.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/numeric/ternary_operator.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric/ternary_operator.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric/ternary_operator.hpp"
#endif

namespace numbirch {

}
