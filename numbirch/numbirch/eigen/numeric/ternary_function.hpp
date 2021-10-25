/**
 * @file
 */
#pragma once

#include "numbirch/numeric/ternary_function.hpp"
#include "numbirch/functor/ternary_function.hpp"
#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

template<class T, class U>
void ibeta(const int m, const int n, const U* A, const int ldA, const U* B,
    const int ldB, const T* X, const int ldX, T* C, const int ldC) {
  ///@todo Implement a generic ternary transform for this purpose
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i + j*ldC] = ibeta(A[i + j*ldA], B[i + j*ldB], X[i + j*ldX]);
    }
  }
}

}
