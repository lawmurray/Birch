/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.hpp"
#endif

#define MATRIX_MATRIX(f) \
    MATRIX_MATRIX_SIG(f, real)
#define MATRIX_MATRIX_SIG(f, T) \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&);

#define MATRIX_MATRIX_GRAD(f) \
    MATRIX_MATRIX_GRAD_SIG(f, real)
#define MATRIX_MATRIX_GRAD_SIG(f, T) \
    template std::pair<Array<T,2>,Array<T,2>> f(const Array<T,2>&, \
        const Array<T,2>&, const Array<T,2>&);

namespace numbirch {
MATRIX_MATRIX(operator*)
MATRIX_MATRIX(trimul)
MATRIX_MATRIX(triouter)
MATRIX_MATRIX(cholsolve)
MATRIX_MATRIX_GRAD(cholsolve_grad)
MATRIX_MATRIX(solve)
}
