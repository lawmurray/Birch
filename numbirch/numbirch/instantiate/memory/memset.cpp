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
    MEMSET_FIRST(f, real) \
    MEMSET_FIRST(f, int) \
    MEMSET_FIRST(f, bool)
#define MEMSET_FIRST(f, T) \
    MEMSET_SIG(f, T, real) \
    MEMSET_SIG(f, T, int) \
    MEMSET_SIG(f, T, bool)
#define MEMSET_SIG(f, T, U) \
    template void f<T,U>(T*, const int, const U, const int, const int); \
    template void f<T,U>(T*, const int, const U*, const int, const int);

namespace numbirch {
MEMSET(memset)
}
