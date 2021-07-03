/**
 * @file
 */
#include "numbirch/numbirch.hpp"

#include <cblas.h>
#include <lapacke.h>

void numbirch::add(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  cblas_dcopy(n, x, incx, z, incz);
  cblas_daxpy(n, 1.0, y, incy, z, incz);
}

void numbirch::add(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  for (int i; i < m; ++i) {
    add(n, A + i*ldA, 1, B + i*ldB, 1, C + i*ldC, 1);
  }
}

void numbirch::sub(const int n, const double* x, const int incx,
    const double* y, const int incy, double* z, const int incz) {
  cblas_dcopy(n, x, incx, z, incz);
  cblas_daxpy(n, -1.0, y, incy, z, incz);
}

void numbirch::sub(const int m, const int n, const double* A, const int ldA,
    const double* B, const int ldB, double* C, const int ldC) {
  for (int i; i < m; ++i) {
    sub(n, A + i*ldA, 1, B + i*ldB, 1, C + i*ldC, 1);
  }
}
