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
    IOTA_SECOND(f, T, int)
#define IOTA_SECOND(f, T, U) \
    IOTA_SIG(f, T, U) \
    IOTA_SIG(f, T, NUMBIRCH_ARRAY(U, 0)) \
    IOTA_SIG(f, NUMBIRCH_ARRAY(T, 0), U) \
    IOTA_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0))
#define IOTA_SIG(f, T, U) \
    template Array<value_t<implicit_t<T,U>>,1> f<T,U,int>(const T& x, \
        const U& y, const int n);

namespace numbirch {
IOTA(iota)
}
