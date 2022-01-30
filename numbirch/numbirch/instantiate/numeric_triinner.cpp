/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.hpp"
#endif

#define TRIINNER(f, f_grad) \
    TRIINNER_SIG(f, f_grad, real)
#define TRIINNER_SIG(f, f_grad, T) \
    template Array<T,1> f(const Array<T,2>&, const Array<T,1>&); \
    template Array<T,2> f(const Array<T,2>&); \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&); \
    template std::pair<Array<T,2>,Array<T,1>> f_grad(const Array<T,1>&, \
        const Array<T,1>&, const Array<T,2>&, const Array<T,1>&); \
    template Array<T,2> f_grad(const Array<T,2>&, const Array<T,2>&, \
        const Array<T,2>&); \
    template std::pair<Array<T,2>,Array<T,2>> f_grad(const Array<T,2>&, \
        const Array<T,2>&, const Array<T,2>&, const Array<T,2>&);

namespace numbirch {
TRIINNER(triinner, triinner_grad)
}
