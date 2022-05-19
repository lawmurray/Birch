/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/memory.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/memory.inl"
#endif

namespace numbirch {

template void memset(void*, const size_t, const double, const size_t,
    const size_t);
template void memset(void*, const size_t, const float, const size_t,
    const size_t);
template void memset(void*, const size_t, const int, const size_t,
    const size_t);
template void memset(void*, const size_t, const bool, const size_t,
    const size_t);

}
