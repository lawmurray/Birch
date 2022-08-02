/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.inl"
#endif

#define DOT(f) \
    DOT_SIG(f, real)
#define DOT_SIG(f, T) \
    template Array<T,0> f(const Array<T,1>&); \
    template Array<T,0> f(const Array<T,1>&, const Array<T,1>&);


namespace numbirch {
DOT(dot)
}
