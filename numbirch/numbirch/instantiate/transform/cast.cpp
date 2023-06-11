/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.inl"
#include "numbirch/cuda/random.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.inl"
#include "numbirch/eigen/random.inl"
#endif
#include "numbirch/common/transform.inl"

#define CAST(f) \
    CAST_FIRST(f, real) \
    CAST_FIRST(f, int) \
    CAST_FIRST(f, bool)
#define CAST_FIRST(f, R) \
    CAST_SECOND(f, R, real) \
    CAST_SECOND(f, R, int) \
    CAST_SECOND(f, R, bool)
#define CAST_SECOND(f, R, T) \
    CAST_SIG(f, R, NUMBIRCH_ARRAY(T, 2)) \
    CAST_SIG(f, R, NUMBIRCH_ARRAY(T, 1)) \
    CAST_SIG(f, R, NUMBIRCH_ARRAY(T, 0)) \
    CAST_SIG(f, R, T)
#define CAST_SIG(f, R, T) \
    template explicit_t<R,T> f<R,T>(const T&);

namespace numbirch {
CAST(cast)
}
