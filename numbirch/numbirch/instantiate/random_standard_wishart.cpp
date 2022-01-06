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
    STANDARD_WISHART_FIRST(f, real)
#define STANDARD_WISHART_FIRST(f, R) \
    STANDARD_WISHART_SECOND(f, R, real) \
    STANDARD_WISHART_SECOND(f, R, int) \
    STANDARD_WISHART_SECOND(f, R, bool)
#define STANDARD_WISHART_SECOND(f, R, T) \
    STANDARD_WISHART_SIG(f, R, ARRAY(T, 0)) \
    STANDARD_WISHART_SIG(f, R, T)
#define STANDARD_WISHART_SIG(f, R, T) \
    template Array<R,2> f<R,T>(const T& nu, const int n);

namespace numbirch {
STANDARD_WISHART(standard_wishart)
}
