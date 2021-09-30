/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/numbirch.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/numbirch.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/numbirch.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/numbirch.hpp"
#endif

namespace numbirch {

template void neg(const int m, const int n, const double* A, const int ldA,
    double* B, const int ldB);
template void neg(const int m, const int n, const float* A, const int ldA,
    float* B, const int ldB);

template void rectify(const int m, const int n, const double* A,
    const int ldA, double* B, const int ldB);
template void rectify(const int m, const int n, const float* A,
    const int ldA, float* B, const int ldB);

template void add(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);
template void add(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC);

template void sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);
template void sub(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC);

template void combine(const int m, const int n, const double a,
    const double* A, const int ldA, const double b, const double* B,
    const int ldB, const double c, const double* C, const int ldC,
    const double d, const double* D, const int ldD, double* E, const int ldE);
template void combine(const int m, const int n, const float a,
    const float* A, const int ldA, const float b, const float* B,
    const int ldB, const float c, const float* C, const int ldC,
    const float d, const float* D, const int ldD, float* E, const int ldE);

template void hadamard(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);
template void hadamard(const int m, const int n, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC);

template void div(const int m, const int n, const double* A, const int ldA,
    const double b, double* C, const int ldC);
template void div(const int m, const int n, const float* A, const int ldA,
    const float b, float* C, const int ldC);

template void mul(const int m, const int n, const double a, const double* B,
    const int ldB, double* C, const int ldC);
template void mul(const int m, const int n, const float a, const float* B,
    const int ldB, float* C, const int ldC);

template void mul(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy);
template void mul(const int m, const int n, const float* A, const int ldA,
    const float* x, const int incx, float* y, const int incy);

template void mul(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);
template void mul(const int m, const int n, const int k, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC);

template void cholmul(const int n, const double* S, const int ldS,
    const double* x, const int incx, double* y, const int incy);
template void cholmul(const int n, const float* S, const int ldS,
    const float* x, const int incx, float* y, const int incy);

template void cholmul(const int m, const int n, const double* S,
    const int ldS, const double* B, const int ldB, double* C, const int ldC);
template void cholmul(const int m, const int n, const float* S,
    const int ldS, const float* B, const int ldB, float* C, const int ldC);

template double sum(const int m, const int n, const double* A, const int ldA);
template float sum(const int m, const int n, const float* A, const int ldA);

template double dot(const int n, const double* x, const int incx,
    const double* y, const int incy);
template float dot(const int n, const float* x, const int incx,
    const float* y, const int incy);

template double frobenius(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB);
template float frobenius(const int m, const int n, const float* A,
    const int ldA, const float* B, const int ldB);

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

template void cholouter(const int m, const int n, const double* A,
    const int ldA, const double* S, const int ldS, double* C, const int ldC);
template void cholouter(const int m, const int n, const float* A,
    const int ldA, const float* S, const int ldS, float* C, const int ldC);

template void solve(const int n, const double* A, const int ldA, double* x,
    const int incx, const double* y, const int incy);
template void solve(const int n, const float* A, const int ldA, float* x,
    const int incx, const float* y, const int incy);

template void solve(const int m, const int n, const double* A, const int ldA,
    double* X, const int ldX, const double* Y, const int ldY);
template void solve(const int m, const int n, const float* A, const int ldA,
    float* X, const int ldX, const float* Y, const int ldY);

template void cholsolve(const int n, const double* S, const int ldS,
    double* x, const int incx, const double* y, const int incy);
template void cholsolve(const int n, const float* S, const int ldS,
    float* x, const int incx, const float* y, const int incy);

template void cholsolve(const int m, const int n, const double* S,
    const int ldS, double* X, const int ldX, const double* Y, const int ldY);
template void cholsolve(const int m, const int n, const float* S,
    const int ldS, float* X, const int ldX, const float* Y, const int ldY);

template void inv(const int n, const double* A, const int ldA, double* B,
    const int ldB);
template void inv(const int n, const float* A, const int ldA, float* B,
    const int ldB);

template void cholinv(const int n, const double* S, const int ldS, double* B,
    const int ldB);
template void cholinv(const int n, const float* S, const int ldS, float* B,
    const int ldB);

template double ldet(const int n, const double* A, const int ldA);
template float ldet(const int n, const float* A, const int ldA);

template double lcholdet(const int n, const double* S, const int ldS);
template float lcholdet(const int n, const float* S, const int ldS);

template void transpose(const int m, const int n, const double x,
    const double* A, const int ldA, double* B, const int ldB);
template void transpose(const int m, const int n, const float x,
    const float* A, const int ldA, float* B, const int ldB);

template double trace(const int m, const int n, const double* A,
    const int ldA);
template float trace(const int m, const int n, const float* A,
    const int ldA);

}
