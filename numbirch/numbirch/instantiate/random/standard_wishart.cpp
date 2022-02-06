/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/random.hpp"
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/random.hpp"
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/random.hpp"

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
    STANDARD_WISHART_SIG(f, ARRAY(T, 0)) \
    STANDARD_WISHART_SIG(f, T)
#define STANDARD_WISHART_SIG(f, T) \
    template Array<real,2> f<T>(const T& nu, const int n);

namespace numbirch {
STANDARD_WISHART(standard_wishart)
}
