/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.inl"
#endif

#define BINARY_MATRIX(f) \
    BINARY_MATRIX_SIG(f, real)
#define BINARY_MATRIX_SIG(f, T) \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&); \
    template Array<T,1> f(const Array<T,2>&, const Array<T,1>&); \
    template Array<T,2> f<T,Array<T,0>,int>(const Array<T,2>&, const Array<T,0>&); \
    template Array<T,2> f<T,T,int>(const Array<T,2>&, const T&);

namespace numbirch {
BINARY_MATRIX(cholsolve)
BINARY_MATRIX(triinnersolve)
BINARY_MATRIX(trisolve)
}
