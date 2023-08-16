/**
 * @file
 */
#include "numbirch/common/numeric.inl"
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.inl"
#endif
#include "numbirch/instantiate/instantiate.hpp"

namespace numbirch {

NUMBIRCH_KEEP static void instantiate() {
  std::visit([]<class T>(T A) {
    transpose(A);

    real_t<T> g;
    transpose_grad(g, A);
  }, matrix_variant());
}

}
