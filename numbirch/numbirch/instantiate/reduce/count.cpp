/**
 * @file
 */
#include "numbirch/common/reduce.inl"
#ifdef BACKEND_CUDA
#include "numbirch/cuda/reduce.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/reduce.inl"
#endif

#define COUNT(f) \
    COUNT_FIRST(f, real) \
    COUNT_FIRST(f, int) \
    COUNT_FIRST(f, bool)
#define COUNT_FIRST(f, T) \
    COUNT_SIG(f, NUMBIRCH_ARRAY(T, 2)) \
    COUNT_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    COUNT_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    COUNT_SIG(f, T)
#define COUNT_SIG(f, T) \
    template Array<int,0> f(const T&);

namespace numbirch {
COUNT(count)
}
