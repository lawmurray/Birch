/**
 * @file
 */
#pragma once

#include "numbirch/oneapi/sycl.hpp"
#include "numbirch/oneapi/mkl.hpp"
#include "numbirch/oneapi/dpl.hpp"
#include "numbirch/jemalloc/jemalloc.hpp"
#include "numbirch/functor.hpp"
#include "numbirch/memory.hpp"

namespace numbirch {

void logical_not(const int m, const int n, const bool* A, const int ldA,
    bool* B, const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(),
      A1.end(), B1.begin(), logical_not_functor());
}

template<class T>
void neg(const int m, const int n, const T* A, const int ldA, T* B,
    const int ldB) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  dpl::transform(dpl::execution::make_device_policy(queue), A1.begin(),
      A1.end(), B1.begin(), dpl::negate<T>());
}

}
