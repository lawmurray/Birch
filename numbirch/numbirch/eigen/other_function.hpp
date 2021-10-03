/**
 * @file
 */
#pragma once

#include "numbirch/numeric/other_function.hpp"
#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

template<class T>
void combine(const int m, const int n, const T a, const T* A, const int ldA,
    const T b, const T* B, const int ldB, const T c, const T* C,
    const int ldC, const T d, const T* D, const int ldD, T* E,
    const int ldE) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  auto C1 = make_eigen_matrix(C, m, n, ldC);
  auto D1 = make_eigen_matrix(D, m, n, ldD);
  auto E1 = make_eigen_matrix(E, m, n, ldE);
  E1.noalias() = a*A1 + b*B1 + c*C1 + d*D1;
}

}
