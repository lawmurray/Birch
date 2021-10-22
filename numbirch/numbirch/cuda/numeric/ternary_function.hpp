/**
 * @file
 */
#pragma once

#include "numbirch/cuda/cuda.hpp"
#include "numbirch/cuda/cublas.hpp"
#include "numbirch/cuda/cusolver.hpp"
#include "numbirch/cuda/cub.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/functor.hpp"
#include "numbirch/memory.hpp"
#include "numbirch/numeric.hpp"

namespace numbirch {

template<class T>
void ibeta(const int m, const int n, const T* A, const int ldA, const T* B,
    const int ldB, const T* X, const int ldX, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(X, m, n, ldX);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, X, ldX, C, ldC, ibeta_functor<T>());
}

template<class T>
void lchoose_grad(const int m, const int n, const T* G, const int ldG,
    const int* A, const int ldA, const int* B, const int ldB, T* GA,
    const int ldGA, T* GB, const int ldGB) {
  prefetch(G, m, n, ldG);
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(GA, m, n, ldGA);
  prefetch(GB, m, n, ldGB);
  transform(m, n, G, ldG, A, ldA, B, ldB, GA, ldGA, GB, ldGB,
      lchoose_grad_functor<T>());
}

}
