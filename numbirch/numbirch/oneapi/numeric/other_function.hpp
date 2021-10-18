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

template<class T>
void combine(const int m, const int n, const T a, const T* A, const int ldA,
    const T b, const T* B, const int ldB, const T c, const T* C,
    const int ldC, const T d, const T* D, const int ldD, T* E,
    const int ldE) {
  auto A1 = make_dpl_matrix(A, m, n, ldA);
  auto B1 = make_dpl_matrix(B, m, n, ldB);
  auto C1 = make_dpl_matrix(C, m, n, ldC);
  auto D1 = make_dpl_matrix(D, m, n, ldD);
  auto E1 = make_dpl_matrix(E, m, n, ldE);

  auto begin = dpl::make_zip_iterator(A1.begin(), B1.begin(), C1.begin(),
      D1.begin());

  dpl::transform(dpl::execution::make_device_policy(queue), begin, begin +
      m*n, E1.begin(), combine4_functor<T>(a, b, c, d));
}

}
