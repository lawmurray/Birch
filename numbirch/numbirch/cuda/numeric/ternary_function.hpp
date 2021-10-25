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

template<class T, class U>
void ibeta(const int m, const int n, const U* A, const int ldA, const U* B,
    const int ldB, const T* X, const int ldX, T* C, const int ldC) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(X, m, n, ldX);
  prefetch(C, m, n, ldC);
  transform(m, n, A, ldA, B, ldB, X, ldX, C, ldC, ibeta_functor<T>());
}

}
