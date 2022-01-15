/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#include "numbirch/cuda/random.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#include "numbirch/eigen/random.hpp"
#endif
#include "numbirch/common/random.hpp"

namespace numbirch {
Array<real,1> standard_gaussian(const int n) {
  return for_each(n, standard_gaussian_functor());
}

Array<real,2> standard_gaussian(const int m, const int n) {
  return for_each(m, n, standard_gaussian_functor());
}
}
