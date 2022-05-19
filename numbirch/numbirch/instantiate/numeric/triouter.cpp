/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.inl"
#endif

#define TRIOUTER(f, f_grad) \
    TRIOUTER_SIG(f, f_grad, real)
#define TRIOUTER_SIG(f, f_grad, T) \
    template Array<T,2> f(const Array<T,2>&); \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&); \
    template Array<T,2> f_grad(const Array<T,2>&, const Array<T,2>&, \
        const Array<T,2>&); \
    template std::pair<Array<T,2>,Array<T,2>> f_grad(const Array<T,2>&, \
        const Array<T,2>&, const Array<T,2>&, const Array<T,2>&);

namespace numbirch {
TRIOUTER(triouter, triouter_grad)
}
