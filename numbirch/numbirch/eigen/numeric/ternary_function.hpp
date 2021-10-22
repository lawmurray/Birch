/**
 * @file
 */
#pragma once

#include "numbirch/numeric/ternary_function.hpp"
#include "numbirch/functor/ternary_function.hpp"
#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

template<class T>
void ibeta(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, const T* X, const int ldX, T* C, const int ldC) {
  ///@todo Implement a generic ternary transform for this purpose
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i + j*ldC] = ibeta(A[i + j*ldA], B[i + j*ldB], X[i + j*ldX]);
    }
  }
}

template<class T>
void lchoose_grad(const int m, const int n, const T* G, const int ldG,
    const int* A, const int ldA, const int* B, const int ldB, T* GA,
    const int ldGA, T* GB, const int ldGB) {
  ///@todo Implement a generic ternary transform for this purpose
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      auto pair = lchoose_grad(G[i + j*ldG], A[i + j*ldA], B[i + j*ldB]);
      GA[i + j*ldGA] = pair.first;
      GB[i + j*ldGB] = pair.second;
    }
  }
}

}
