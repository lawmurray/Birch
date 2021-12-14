/**
 * @file
 */
#pragma once

#include "numbirch/macro.hpp"

namespace numbirch {
/**
 * @internal
 *
 * 0-based element of a matrix, vector, or scalar. A scalar is identified by
 * having `ld == 0`.
 */
template<class T>
NUMBIRCH_HOST_DEVICE T& element(T* x, const int i = 0, const int j = 0,
    const int ld = 0) {
  int k = (ld == 0) ? 0 : (i + j*int64_t(ld));
  return x[k];
}

/**
 * @internal
 *
 * 0-based element of a matrix, vector, or scalar. A scalar is identified by
 * having `ld == 0`.
 */
template<class T>
NUMBIRCH_HOST_DEVICE const T& element(const T* x, const int i = 0,
    const int j = 0, const int ld = 0) {
  int k = (ld == 0) ? 0 : (i + j*int64_t(ld));
  return x[k];
}

/**
 * @internal
 * 
 * 0-based element of a scalar---just returns the scalar.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
NUMBIRCH_HOST_DEVICE T& element(T& x, const int i = 0, const int j = 0,
    const int ld = 0) {
  return x;
}

/**
 * @internal
 * 
 * 0-based element of a scalar---just returns the scalar.
 */
template<class T, class = std::enable_if_t<is_arithmetic_v<T>,int>>
NUMBIRCH_HOST_DEVICE const T& element(const T& x, const int i = 0,
    const int j = 0, const int ld = 0) {
  return x;
}

}