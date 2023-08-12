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

#define DIAGONAL(f) \
    DIAGONAL_FIRST(f, real) \
    DIAGONAL_FIRST(f, int) \
    DIAGONAL_FIRST(f, bool)
#define DIAGONAL_FIRST(f, T) \
    DIAGONAL_SIG(f, T) \
    DIAGONAL_SIG(f, NUMBIRCH_ARRAY(T, 0))
#define DIAGONAL_SIG(f, T) \
    template Array<value_t<T>,2> f<T>(const T& x, const int n); \
    template Array<real,0> f##_grad(const Array<real,2>& g, \
        const T& x, const int n);

namespace numbirch {
DIAGONAL(diagonal)
}
