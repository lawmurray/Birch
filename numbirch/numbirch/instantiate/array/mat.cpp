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

#define MAT(f) \
    MAT_FIRST(f, real) \
    MAT_FIRST(f, int) \
    MAT_FIRST(f, bool)
#define MAT_FIRST(f, T) \
    MAT_SIG(f, T) \
    MAT_SIG(f, NUMBIRCH_ARRAY(T, 0)) \
    MAT_SIG(f, NUMBIRCH_ARRAY(T, 1)) \
    MAT_SIG(f, NUMBIRCH_ARRAY(T, 2))
#define MAT_SIG(f, T) \
    template Array<value_t<T>,2> f<T>(const T& x, const int n); \
    template real_t<T> f##_grad(const Array<real,2>& g, \
        const T& x, const int n);

namespace numbirch {
MAT(mat)
}
