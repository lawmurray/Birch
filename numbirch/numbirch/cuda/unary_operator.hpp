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

void logical_not(const int m, const int n, const bool* A, const int ldA,
    bool* B, const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, logical_not_functor());
}

template<class T>
void neg(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  prefetch(A, m, n, ldA);
  prefetch(B, m, n, ldB);
  transform(m, n, A, ldA, B, ldB, negate_functor<T>());
}

}
