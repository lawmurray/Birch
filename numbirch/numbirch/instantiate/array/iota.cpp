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

#define IOTA(f) \
    IOTA_FIRST(f, real) \
    IOTA_FIRST(f, int) \
    IOTA_FIRST(f, bool)
#define IOTA_FIRST(f, T) \
    IOTA_SIG(f, T) \
    IOTA_SIG(f, NUMBIRCH_ARRAY(T, 0))
#define IOTA_SIG(f, T) \
    template Array<value_t<T>,1> f<T,int>(const T& x, const int n); \
    template Array<real,0> f##_grad<T,int>(const Array<real,1>& g, \
        const Array<value_t<T>,1>& y, const T& x, const int n);

namespace numbirch {
IOTA(iota)
}
