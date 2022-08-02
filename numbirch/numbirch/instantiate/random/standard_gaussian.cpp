/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#include "numbirch/cuda/random.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#include "numbirch/eigen/random.inl"
#endif
#include "numbirch/common/random.inl"

namespace numbirch {
Array<real,1> standard_gaussian(const int n) {
  return for_each(n, standard_gaussian_functor());
}

Array<real,2> standard_gaussian(const int m, const int n) {
  return for_each(m, n, standard_gaussian_functor());
}
}
