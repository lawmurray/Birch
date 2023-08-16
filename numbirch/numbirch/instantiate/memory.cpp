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

NUMBIRCH_KEEP void instantiate2(int a, int b, int c, int d) {
  std::visit([=]<class T, class U>(T x, U y) {
    // memcpy(&x, a, &y, b, c, d);
    // memset(&x, a, &y, b, c);
    // memset(&x, a, y, b, c);
  }, arithmetic_variant(), arithmetic_variant());
}

template void memset<real>(real*, int, real, int, int);
template void memset<real>(real*, int, const real*, int, int);
template void memset<int>(int*, int, int, int, int);
template void memset<int>(int*, int, const int*, int, int);
template void memset<bool>(bool*, int, bool, int, int);
template void memset<bool>(bool*, int, const bool*, int, int);

template void memcpy<real,real>(real*, int, const real*, int, int, int);
template void memcpy<real,int>(real*, int, const int*, int, int, int);
template void memcpy<real,bool>(real*, int, const bool*, int, int, int);
template void memcpy<int,real>(int*, int, const real*, int, int, int);
template void memcpy<int,int>(int*, int, const int*, int, int, int);
template void memcpy<int,bool>(int*, int, const bool*, int, int, int);
template void memcpy<bool,real>(bool*, int, const real*, int, int, int);
template void memcpy<bool,int>(bool*, int, const int*, int, int, int);
template void memcpy<bool,bool>(bool*, int, const bool*, int, int, int);

}
