/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/numeric/ternary_operator.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/ternary_operator.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/ternary_operator.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/ternary_operator.hpp"
#endif

namespace numbirch {

}
