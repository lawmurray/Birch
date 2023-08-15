/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/memory.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/memory.inl"
#endif
#include "numbirch/instantiate/instantiate.hpp"

namespace numbirch {

[[maybe_unused]] static void instantiate() {
  std::visit([]<class T, class U>(T x, U y) {
    memcpy(&x, 0, &y, 0, 0, 0);
    memset(&x, 0, &y, 0, 0);
    memset(&x, 0, y, 0, 0);
  }, arithmetic_variant(), arithmetic_variant());
}

}
