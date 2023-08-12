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

#define PACK(f) \
    PACK_FIRST(f, real) \
    PACK_FIRST(f, int) \
    PACK_FIRST(f, bool)
#define PACK_FIRST(f, T) \
    PACK_SECOND(f, T, real) \
    PACK_SECOND(f, T, int) \
    PACK_SECOND(f, T, bool)
#define PACK_SECOND(f, T, U) \
    PACK_SIG(f, T, U) \
    PACK_SIG(f, T, NUMBIRCH_ARRAY(U, 0)) \
    PACK_SIG(f, T, NUMBIRCH_ARRAY(U, 1)) \
    PACK_SIG(f, T, NUMBIRCH_ARRAY(U, 2)) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 0), U) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0)) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 1)) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 2)) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 1), U) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 0)) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 1)) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 1), NUMBIRCH_ARRAY(U, 2)) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 2), U) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 0)) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 1)) \
    PACK_SIG(f, NUMBIRCH_ARRAY(T, 2), NUMBIRCH_ARRAY(U, 2))

#define PACK_SIG(f, T, U) \
    template pack_t<T,U> f(const T& x, const U& y); \
    template real_t<T> f##_grad1(const real_t<pack_t<T,U>>& g, \
        const T& x, const U& y); \
    template real_t<U> f##_grad2(const real_t<pack_t<T,U>>& g, \
        const T& x, const U& y);

namespace numbirch {
PACK(pack)
}
