/**
 * @file
 */
#include "numbirch/random.hpp"

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
    STANDARD_WISHART_FIRST(f, double) \
    STANDARD_WISHART_FIRST(f, float)
#define STANDARD_WISHART_FIRST(f, R) \
    STANDARD_WISHART_SIG(f, R, double) \
    STANDARD_WISHART_SIG(f, R, float) \
    STANDARD_WISHART_SIG(f, R, int) \
    STANDARD_WISHART_SIG(f, R, bool) \
    STANDARD_WISHART_SIG(f, R, ARRAY(double, 0)) \
    STANDARD_WISHART_SIG(f, R, ARRAY(float, 0)) \
    STANDARD_WISHART_SIG(f, R, ARRAY(int, 0)) \
    STANDARD_WISHART_SIG(f, R, ARRAY(bool, 0))
#define STANDARD_WISHART_SIG(f, R, T) \
    template Array<R,2> f<R,T>(const T& nu, const int n);

namespace numbirch {
STANDARD_WISHART(standard_wishart)
}
