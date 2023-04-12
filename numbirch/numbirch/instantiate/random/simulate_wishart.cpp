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
 * @def SIMULATE_WISHART
 */
#define SIMULATE_WISHART(f) \
    SIMULATE_WISHART_FIRST(f, real) \
    SIMULATE_WISHART_FIRST(f, int) \
    SIMULATE_WISHART_FIRST(f, bool)
#define SIMULATE_WISHART_FIRST(f, T) \
    SIMULATE_WISHART_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    SIMULATE_WISHART_SIG(f, T)
#define SIMULATE_WISHART_SIG(f, T) \
    template Array<real,2> f<T>(const T& nu, const int n);

namespace numbirch {
SIMULATE_WISHART(simulate_wishart)
}
