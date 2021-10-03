/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/numeric/unary_function.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/unary_function.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/unary_function.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/unary_function.hpp"
#endif

namespace numbirch {

template void cholinv(const int n, const double* S, const int ldS, double* B,
    const int ldB);
template void cholinv(const int n, const float* S, const int ldS, float* B,
    const int ldB);

template void inv(const int n, const double* A, const int ldA, double* B,
    const int ldB);
template void inv(const int n, const float* A, const int ldA, float* B,
    const int ldB);

template void lcholdet(const int n, const double* S, const int ldS,
    double* b);
template void lcholdet(const int n, const float* S, const int ldS,
    float* b);

template void ldet(const int n, const double* A, const int ldA, double* b);
template void ldet(const int n, const float* A, const int ldA, float* b);

template void rectify(const int m, const int n, const double* A,
    const int ldA, double* B, const int ldB);
template void rectify(const int m, const int n, const float* A,
    const int ldA, float* B, const int ldB);

template void sum(const int m, const int n, const double* A, const int ldA,
    double* b);
template void sum(const int m, const int n, const float* A, const int ldA,
    float* b);
template void sum(const int m, const int n, const int* A, const int ldA,
    int* b);

template void trace(const int m, const int n, const double* A, const int ldA,
    double* b);
template void trace(const int m, const int n, const float* A, const int ldA,
    float* b);

template void transpose(const int m, const int n, const double* A,
    const int ldA, double* B, const int ldB);
template void transpose(const int m, const int n, const float* A,
    const int ldA, float* B, const int ldB);

}
