/**
 * @file
 */
#pragma once

#include "numbirch/utility.hpp"

namespace numbirch {

template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP void memcpy(T* dst, const int dpitch, const U* src, const int spitch,
    const int width, const int height) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      get(dst, i, j, dpitch) = get(src, i, j, spitch);
    }
  }
}

template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP void memset(T* dst, const int dpitch, const U value, const int width,
    const int height) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      get(dst, i, j, dpitch) = value;
    }
  }
}

template<arithmetic T, arithmetic U>
NUMBIRCH_KEEP void memset(T* dst, const int dpitch, const U* value, const int width,
    const int height) {
  for (int j = 0; j < height; ++j) {
    for (int i = 0; i < width; ++i) {
      get(dst, i, j, dpitch) = get(value);
    }
  }
}

}
