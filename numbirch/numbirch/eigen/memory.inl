/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"

namespace numbirch {

template<class T, class>
void memset(void* dst, const size_t dpitch, const T value, const size_t width,
    const size_t height) {
  auto A = (T*)dst;
  auto ld = dpitch/sizeof(T);
  auto m = width/sizeof(T);
  auto n = height;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      A[i + j*ld] = value;
    }
  }
}

}
