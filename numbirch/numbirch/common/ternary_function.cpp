/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/numeric/ternary_function.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/ternary_function.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/ternary_function.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/ternary_function.hpp"
#endif

namespace numbirch {

}
