/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/random.inl"
#include "numbirch/cuda/transform.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/random.inl"
#include "numbirch/eigen/transform.inl"
#endif
#include "numbirch/common/random.inl"

/**
 * @internal
 * 
 * @def STANDARD_WISHART
 */
#define STANDARD_WISHART(f) \
    STANDARD_WISHART_FIRST(f, real) \
    STANDARD_WISHART_FIRST(f, int) \
    STANDARD_WISHART_FIRST(f, bool)
#define STANDARD_WISHART_FIRST(f, T) \
    STANDARD_WISHART_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    STANDARD_WISHART_SIG(f, T)
#define STANDARD_WISHART_SIG(f, T) \
    template Array<real,2> f<T>(const T& nu, const int n);

namespace numbirch {
STANDARD_WISHART(standard_wishart)
}
