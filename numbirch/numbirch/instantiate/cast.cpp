/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#include "numbirch/cuda/random.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#include "numbirch/eigen/random.hpp"
#endif
#include "numbirch/common/unary.hpp"

#define CAST(f) \
    CAST_FIRST(f, real) \
    CAST_FIRST(f, int) \
    CAST_FIRST(f, bool)
#define CAST_FIRST(f, R) \
    CAST_SECOND(f, R, real) \
    CAST_SECOND(f, R, int) \
    CAST_SECOND(f, R, bool)
#define CAST_SECOND(f, R, T) \
    CAST_SIG(f, R, ARRAY(T, 2)) \
    CAST_SIG(f, R, ARRAY(T, 1)) \
    CAST_SIG(f, R, ARRAY(T, 0)) \
    CAST_SIG(f, R, T)
#define CAST_SIG(f, R, T) \
    template explicit_t<R,T> f<R,T,int>(const T&);

namespace numbirch {
CAST(cast)
}
