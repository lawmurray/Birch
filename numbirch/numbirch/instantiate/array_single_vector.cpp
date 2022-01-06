/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/transform.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/transform.hpp"
#endif
#include "numbirch/common/array.hpp"

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
    SINGLE_VECTOR_SIG(f, T, ARRAY(U, 0)) \
    SINGLE_VECTOR_SIG(f, ARRAY(T, 0), U) \
    SINGLE_VECTOR_SIG(f, ARRAY(T, 0), ARRAY(U, 0))
#define SINGLE_VECTOR_SIG(f, T, U) \
    template Array<value_t<T>,1> f<T,U,int>(const T& x, const U& i, \
        const int n);

namespace numbirch {
SINGLE_VECTOR(single)
}
