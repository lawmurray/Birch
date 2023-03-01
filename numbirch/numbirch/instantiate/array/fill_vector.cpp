/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#endif
#include "numbirch/common/array.inl"

#define FILL(f) \
    FILL_FIRST(f, real) \
    FILL_FIRST(f, int) \
    FILL_FIRST(f, bool)
#define FILL_FIRST(f, T) \
    FILL_SIG(f, T) \
    FILL_SIG(f, NUMBIRCH_ARRAY(T, 0))
#define FILL_SIG(f, T) \
    template Array<value_t<T>,1> f<T,int>(const T& x, const int n);

namespace numbirch {
FILL(fill)
FILL(iota)
}
