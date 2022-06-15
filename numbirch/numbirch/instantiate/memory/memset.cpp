/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/memory.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/memory.inl"
#endif

#define MEMSET(f) \
    MEMSET_SIG(f, real) \
    MEMSET_SIG(f, int) \
    MEMSET_SIG(f, bool)
#define MEMSET_SIG(f, T) \
    template void f<T,int>(T*, const int, const T, const int, const int);

namespace numbirch {
MEMSET(memset)
}
