/**
 * @file
 */
#pragma once

#include "numbirch/memory.hpp"
#include "numbirch/utility.hpp"

namespace numbirch {

template<class T, class U, class>
void memcpy(T* dst, const int dpitch, const U* src, const int spitch,
    const int width, const int height) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      get(dst, i, j, dpitch) = get(src, i, j, spitch);
    }
  }
}

template<class T, class U, class>
void memset(T* dst, const int dpitch, const U value, const int width,
    const int height) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      get(dst, i, j, dpitch) = value;
    }
  }
}

template<class T, class U, class>
void memset(T* dst, const int dpitch, const U* value, const int width,
    const int height) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      get(dst, i, j, dpitch) = get(value);
    }
  }
}

}
