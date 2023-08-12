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
 * @def ELEMENT_VECTOR
 * 
 * For element().
 */
#define ELEMENT_VECTOR(f) \
    ELEMENT_VECTOR_FIRST(f, real) \
    ELEMENT_VECTOR_FIRST(f, int) \
    ELEMENT_VECTOR_FIRST(f, bool)
#define ELEMENT_VECTOR_FIRST(f, T) \
    ELEMENT_VECTOR_SECOND(f, T, int)
#define ELEMENT_VECTOR_SECOND(f, T, U) \
    ELEMENT_VECTOR_SIG(f, T, U) \
    ELEMENT_VECTOR_SIG(f, T, NUMBIRCH_ARRAY(U, 0))
#define ELEMENT_VECTOR_SIG(f, T, U) \
    template Array<T,0> f<T,U>(const Array<T,1>& x, const U& i); \
    template Array<real,1> f##_grad1(const Array<real,0>& g, \
        const Array<T,1>& x, const U& i); \
    template real f##_grad2(const Array<real,0>& g, \
        const Array<T,1>& x, const U& i);

namespace numbirch {
ELEMENT_VECTOR(element)
}
