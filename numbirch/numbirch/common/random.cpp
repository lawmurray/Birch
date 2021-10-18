/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/random.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/random.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/random.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/random.hpp"
#endif

namespace numbirch {

}
