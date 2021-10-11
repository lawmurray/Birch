/**
 * @file
 * 
 * Explicit instantiations of numeric functions for the enabled backend.
 */
#include "numbirch/numeric/binary_operator.hpp"

#ifdef BACKEND_ONEAPI
#include "numbirch/oneapi/binary_operator.hpp"
#endif
#ifdef BACKEND_CUDA
#include "numbirch/cuda/binary_operator.hpp"
#endif
#ifdef BACKEND_EIGEN
#include "numbirch/eigen/binary_operator.hpp"
#endif

namespace numbirch {

template void add(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);
template void add(const int m, const int n, const double* A, const int ldA,
    const float* B, const int ldB, double* C, const int ldC);
template void add(const int m, const int n, const double* A, const int ldA,
    const int* B, const int ldB, double* C, const int ldC);
template void add(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC);
template void add(const int m, const int n, const float* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);
template void add(const int m, const int n, const float* A, const int ldA,
    const int* B, const int ldB, float* C, const int ldC);
template void add(const int m, const int n, const int* A, const int ldA,
    const int* B, const int ldB, int* C, const int ldC);
template void add(const int m, const int n, const int* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC);
template void add(const int m, const int n, const int* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);

template void div(const int m, const int n, const double* A, const int ldA,
    const double* b, double* C, const int ldC);
template void div(const int m, const int n, const double* A, const int ldA,
    const float* b, double* C, const int ldC);
template void div(const int m, const int n, const double* A, const int ldA,
    const int* b, double* C, const int ldC);
template void div(const int m, const int n, const float* A, const int ldA,
    const float* b, float* C, const int ldC);
template void div(const int m, const int n, const float* A, const int ldA,
    const double* b, double* C, const int ldC);
template void div(const int m, const int n, const float* A, const int ldA,
    const int* b, float* C, const int ldC);
template void div(const int m, const int n, const int* A, const int ldA,
    const int* b, int* C, const int ldC);
template void div(const int m, const int n, const int* A, const int ldA,
    const float* b, float* C, const int ldC);
template void div(const int m, const int n, const int* A, const int ldA,
    const double* b, double* C, const int ldC);

template void equal(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, bool* C, const int ldC);
template void equal(const int m, const int n, const double* A, const int ldA,
    const float* B, const int ldB, bool* C, const int ldC);
template void equal(const int m, const int n, const double* A, const int ldA,
    const int* B, const int ldB, bool* C, const int ldC);
template void equal(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, bool* C, const int ldC);
template void equal(const int m, const int n, const float* A, const int ldA,
    const double* B, const int ldB, bool* C, const int ldC);
template void equal(const int m, const int n, const float* A, const int ldA,
    const int* B, const int ldB, bool* C, const int ldC);
template void equal(const int m, const int n, const int* A, const int ldA,
    const int* B, const int ldB, bool* C, const int ldC);
template void equal(const int m, const int n, const int* A, const int ldA,
    const float* B, const int ldB, bool* C, const int ldC);
template void equal(const int m, const int n, const int* A, const int ldA,
    const double* B, const int ldB, bool* C, const int ldC);

template void greater(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void greater(const int m, const int n, const double* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void greater(const int m, const int n, const double* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void greater(const int m, const int n, const float* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void greater(const int m, const int n, const float* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void greater(const int m, const int n, const float* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void greater(const int m, const int n, const int* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void greater(const int m, const int n, const int* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void greater(const int m, const int n, const int* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);

template void greater_or_equal(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void greater_or_equal(const int m, const int n, const double* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void greater_or_equal(const int m, const int n, const double* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void greater_or_equal(const int m, const int n, const float* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void greater_or_equal(const int m, const int n, const float* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void greater_or_equal(const int m, const int n, const float* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void greater_or_equal(const int m, const int n, const int* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void greater_or_equal(const int m, const int n, const int* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void greater_or_equal(const int m, const int n, const int* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);

template void less(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void less(const int m, const int n, const double* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void less(const int m, const int n, const double* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void less(const int m, const int n, const float* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void less(const int m, const int n, const float* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void less(const int m, const int n, const float* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void less(const int m, const int n, const int* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void less(const int m, const int n, const int* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void less(const int m, const int n, const int* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);

template void less_or_equal(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void less_or_equal(const int m, const int n, const double* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void less_or_equal(const int m, const int n, const double* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void less_or_equal(const int m, const int n, const float* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void less_or_equal(const int m, const int n, const float* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void less_or_equal(const int m, const int n, const float* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void less_or_equal(const int m, const int n, const int* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void less_or_equal(const int m, const int n, const int* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void less_or_equal(const int m, const int n, const int* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);

void logical_and(const int m, const int n, const bool* A, const int ldA,
    const bool* B, const int ldB, bool* C, const int ldC);

void logical_or(const int m, const int n, const bool* A, const int ldA,
    const bool* B, const int ldB, bool* C, const int ldC);

template void mul(const int m, const int n, const double* a, const double* B,
    const int ldB, double* C, const int ldC);
template void mul(const int m, const int n, const float* a, const double* B,
    const int ldB, double* C, const int ldC);
template void mul(const int m, const int n, const int* a, const double* B,
    const int ldB, double* C, const int ldC);
template void mul(const int m, const int n, const float* a, const float* B,
    const int ldB, float* C, const int ldC);
template void mul(const int m, const int n, const double* a, const float* B,
    const int ldB, double* C, const int ldC);
template void mul(const int m, const int n, const int* a, const float* B,
    const int ldB, float* C, const int ldC);
template void mul(const int m, const int n, const int* a, const int* B,
    const int ldB, int* C, const int ldC);
template void mul(const int m, const int n, const double* a, const int* B,
    const int ldB, double* C, const int ldC);
template void mul(const int m, const int n, const float* a, const int* B,
    const int ldB, float* C, const int ldC);

template void mul(const int m, const int n, const double* A, const int ldA,
    const double* x, const int incx, double* y, const int incy);
template void mul(const int m, const int n, const float* A, const int ldA,
    const float* x, const int incx, float* y, const int incy);

template void mul(const int m, const int n, const int k, const double* A,
    const int ldA, const double* B, const int ldB, double* C, const int ldC);
template void mul(const int m, const int n, const int k, const float* A,
    const int ldA, const float* B, const int ldB, float* C, const int ldC);

template void not_equal(const int m, const int n, const double* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void not_equal(const int m, const int n, const double* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void not_equal(const int m, const int n, const double* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void not_equal(const int m, const int n, const float* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void not_equal(const int m, const int n, const float* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);
template void not_equal(const int m, const int n, const float* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void not_equal(const int m, const int n, const int* A,
    const int ldA, const int* B, const int ldB, bool* C, const int ldC);
template void not_equal(const int m, const int n, const int* A,
    const int ldA, const float* B, const int ldB, bool* C, const int ldC);
template void not_equal(const int m, const int n, const int* A,
    const int ldA, const double* B, const int ldB, bool* C, const int ldC);

template void sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);
template void sub(const int m, const int n, const double* A, const int ldA,
    const float* B, const int ldB, double* C, const int ldC);
template void sub(const int m, const int n, const double* A, const int ldA,
    const int* B, const int ldB, double* C, const int ldC);
template void sub(const int m, const int n, const float* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC);
template void sub(const int m, const int n, const float* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);
template void sub(const int m, const int n, const float* A, const int ldA,
    const int* B, const int ldB, float* C, const int ldC);
template void sub(const int m, const int n, const int* A, const int ldA,
    const int* B, const int ldB, int* C, const int ldC);
template void sub(const int m, const int n, const int* A, const int ldA,
    const float* B, const int ldB, float* C, const int ldC);
template void sub(const int m, const int n, const int* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC);

}
