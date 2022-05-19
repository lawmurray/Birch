/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"

namespace numbirch {

template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int>>
void memset(void* dst, const size_t dpitch, const T value, const size_t width,
    const size_t height) {
  auto A = (T*)dst;
  for (int i = 0; i < width/sizeof(T); ++i) {
    for (int j = 0; j < height; ++j) {
      A[i + j*dpitch/sizeof(T)] = value;
    }
  }
}

}
