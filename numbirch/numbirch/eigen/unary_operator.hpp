/**
 * @file
 */
#pragma once

#include "numbirch/numeric/unary_operator.hpp"
#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

template<class T>
void neg(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_eigen_matrix(A, m, n, ldA);
  auto B1 = make_eigen_matrix(B, m, n, ldB);
  B1.noalias() = -A1;
}

}
