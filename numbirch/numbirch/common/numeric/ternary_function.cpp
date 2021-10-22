/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/numeric/ternary_function.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/numeric/ternary_function.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numeric/ternary_function.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numeric/ternary_function.hpp"
#endif

namespace numbirch {

template void ibeta(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, const double* X, const int ldX, double* C,
    const int ldC);
template void ibeta(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, const float* X, const int ldX, float* C,
    const int ldC);

template void lchoose_grad(const int m, const int n, const double* G,
    const int ldG, const int* A, const int ldA, const int* B,
    const int ldB, double* GA, const int ldGA, double* GB, const int ldGB);
template void lchoose_grad(const int m, const int n, const float* G,
    const int ldG, const int* A, const int ldA, const int* B,
    const int ldB, float* GA, const int ldGA, float* GB, const int ldGB);

}
