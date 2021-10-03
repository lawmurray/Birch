/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/numeric/binary_function.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/binary_function.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/binary_function.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/binary_function.hpp"
#endif

namespace numbirch {
template void cholmul(const int n, const double* S, const int ldS,
    const double* x, const int incx, double* y, const int incy);
template void cholmul(const int n, const float* S, const int ldS,
    const float* x, const int incx, float* y, const int incy);

template void cholmul(const int m, const int n, const double* S,
    const int ldS, const double* B, const int ldB, double* C, const int ldC);
template void cholmul(const int m, const int n, const float* S,
    const int ldS, const float* B, const int ldB, float* C, const int ldC);

template void cholouter(const int m, const int n, const double* A,
    const int ldA, const double* S, const int ldS, double* C, const int ldC);
template void cholouter(const int m, const int n, const float* A,
    const int ldA, const float* S, const int ldS, float* C, const int ldC);

template void cholsolve(const int n, const double* S, const int ldS,
    double* x, const int incx, const double* y, const int incy);
template void cholsolve(const int n, const float* S, const int ldS,
    float* x, const int incx, const float* y, const int incy);

template void cholsolve(const int m, const int n, const double* S,
    const int ldS, double* X, const int ldX, const double* Y, const int ldY);
template void cholsolve(const int m, const int n, const float* S,
    const int ldS, float* X, const int ldX, const float* Y, const int ldY);

template void diagonal(const double* a, const int n, double* B,
    const int ldB);
template void diagonal(const float* a, const int n, float* B,
    const int ldB);

template void dot(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z);
template void dot(const int n, const float* x, const int incx,
    const float* y, const int incy, float* z);

template void frobenius(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, double* c);
template void frobenius(const int m, const int n, const float* A,
    const int ldA, const float* B, const int ldB, float* c);

template void hadamard(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);
template void hadamard(const int m, const int n, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC);

template void inner(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy);
template void inner(const int m, const int n, const float* A, const int ldA,
    const float* x, const int incx, float* y, const int incy);

template void inner(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);
template void inner(const int m, const int n, const int k, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC);

template void outer(const int m, const int n, const double* x, const int incx,
    const double* y, const int incy, double* A, const int ldA);
template void outer(const int m, const int n, const float* x, const int incx,
    const float* y, const int incy, float* A, const int ldA);

template void outer(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);
template void outer(const int m, const int n, const int k, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC);

template void solve(const int n, const double* A, const int ldA, double* x,
    const int incx, const double* y, const int incy);
template void solve(const int n, const float* A, const int ldA, float* x,
    const int incx, const float* y, const int incy);

template void solve(const int m, const int n, const double* A, const int ldA,
    double* X, const int ldX, const double* Y, const int ldY);
template void solve(const int m, const int n, const float* A, const int ldA,
    float* X, const int ldX, const float* Y, const int ldY);

}
