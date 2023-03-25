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

#define VEC(f) \
    VEC_FIRST(f, real) \
    VEC_FIRST(f, int) \
    VEC_FIRST(f, bool)
#define VEC_FIRST(f, T) \
    VEC_SIG(f, T) \
    VEC_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    VEC_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    VEC_SIG(f, NUMBIRCH_ARRAY(T, 2))
#define VEC_SIG(f, T) \
    template Array<value_t<T>,1> f<T,int>(const T& x); \
    template real_t<T> f##_grad(const Array<real,1>& g, \
        const Array<value_t<T>,1>& y, const T& x);
namespace numbirch {
VEC(vec)
}
