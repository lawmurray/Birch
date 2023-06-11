/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/memory.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/memory.inl"
#endif

#define MEMCPY(f) \
    MEMCPY_FIRST(f, real) \
    MEMCPY_FIRST(f, int) \
    MEMCPY_FIRST(f, bool)
#define MEMCPY_FIRST(f, T) \
    MEMCPY_SIG(f, T, real) \
    MEMCPY_SIG(f, T, int) \
    MEMCPY_SIG(f, T, bool)
#define MEMCPY_SIG(f, T, U) \
    template void f<T,U>(T*, const int, const U*, const int, const int, \
        const int);

namespace numbirch {
MEMCPY(memcpy)
}
