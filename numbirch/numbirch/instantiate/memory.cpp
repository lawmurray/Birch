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

NUMBIRCH_KEEP static void instantiate(int a, int b, int c, int d) {
  std::visit([=]<class T, class U>(T x, U y) {
    memcpy(&x, a, &y, b, c, d);
    memset(&x, a, &y, b, c);
    memset(&x, a, y, b, c);
  }, arithmetic_variant(), arithmetic_variant());
}

}
