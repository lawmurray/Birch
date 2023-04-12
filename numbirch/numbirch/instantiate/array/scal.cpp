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

#define SCAL(f) \
    SCAL_FIRST(f, real) \
    SCAL_FIRST(f, int) \
    SCAL_FIRST(f, bool)
#define SCAL_FIRST(f, T) \
    SCAL_SIG(f, T) \
    SCAL_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    SCAL_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    SCAL_SIG(f, NUMBIRCH_ARRAY(T, 2))
#define SCAL_SIG(f, T) \
    template Array<value_t<T>,0> f<T,int>(const T& x); \
    template real_t<T> f##_grad(const Array<real,0>& g, \
        const Array<value_t<T>,0>& y, const T& x);
namespace numbirch {
SCAL(scal)
}
