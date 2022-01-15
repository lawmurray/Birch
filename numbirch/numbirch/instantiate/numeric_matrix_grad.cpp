/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.hpp"
#endif

#define MATRIX_GRAD(f) \
    MATRIX_GRAD_SIG(f, real)
#define MATRIX_GRAD_SIG(f, T) \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&, \
        const Array<T,2>&);

namespace numbirch {
MATRIX_GRAD(chol_grad)
MATRIX_GRAD(cholinv_grad)
}
