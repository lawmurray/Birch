/**
 * @file
 */
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric.inl"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric.inl"
#endif

#define BINARY_MATRIX(f, f_grad) \
    BINARY_MATRIX_SIG(f, f_grad, real)
#define BINARY_MATRIX_SIG(f, f_grad, T) \
    template Array<T,2> f(const Array<T,2>&, const Array<T,2>&); \
    template Array<T,1> f(const Array<T,2>&, const Array<T,1>&); \
    template std::pair<Array<T,2>,Array<T,2>> f_grad(const Array<T,2>&, \
        const Array<T,2>&, const Array<T,2>&, const Array<T,2>&); \
    template std::pair<Array<T,2>,Array<T,1>> f_grad(const Array<T,1>&, \
        const Array<T,1>&, const Array<T,2>&, const Array<T,1>&);

namespace numbirch {
BINARY_MATRIX(operator*, op_mul_grad)
BINARY_MATRIX(trimul, trimul_grad)
BINARY_MATRIX(cholsolve, cholsolve_grad)
BINARY_MATRIX(trisolve, trisolve_grad)
}
