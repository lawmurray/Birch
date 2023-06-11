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

/**
 * @internal
 * 
 * @def SINGLE_VECTOR
 * 
 * For single().
 */
#define SINGLE_VECTOR(f) \
    SINGLE_VECTOR_FIRST(f, real) \
    SINGLE_VECTOR_FIRST(f, int) \
    SINGLE_VECTOR_FIRST(f, bool)
#define SINGLE_VECTOR_FIRST(f, T) \
    SINGLE_VECTOR_SECOND(f, T, int)
#define SINGLE_VECTOR_SECOND(f, T, U) \
    SINGLE_VECTOR_SIG(f, T, U) \
    SINGLE_VECTOR_SIG(f, T, NUMBIRCH_ARRAY(U, 0)) \
    SINGLE_VECTOR_SIG(f, NUMBIRCH_ARRAY(T, 0), U) \
    SINGLE_VECTOR_SIG(f, NUMBIRCH_ARRAY(T, 0), NUMBIRCH_ARRAY(U, 0))
#define SINGLE_VECTOR_SIG(f, T, U) \
    template Array<value_t<T>,1> f<T,U>(const T& x, const U& i, \
        const int n); \
    template Array<real,0> f##_grad1(const Array<real,1>& g, \
        const Array<value_t<T>,1>& y, const T& x, const U& i, const int n); \
    template real f##_grad2(const Array<real,1>& g, \
        const Array<value_t<T>,1>& y, const T& x, const U& i, const int n);

namespace numbirch {
SINGLE_VECTOR(single)
}
