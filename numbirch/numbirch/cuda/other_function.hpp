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
void combine(const int m, const int n, const T a, const T* A, const int ldA,
    const T b, const T* B, const int ldB, const T c, const T* C,
    const int ldC, const T d, const T* D, const int ldD, T* E,
    const int ldE) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  prefetch(C, m, n, ldC);
  prefetch(D, m, n, ldD);
  prefetch(E, m, n, ldE);

  transform(m, n, A, ldA, B, ldB, C, ldC, D, ldD, E, ldE,
      combine_functor<T>(a, b, c, d));
}

}
