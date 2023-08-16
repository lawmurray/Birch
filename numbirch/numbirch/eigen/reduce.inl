/**
 * @file
 */
#pragma once

#include "numbirch/reduce.hpp"
#include "numbirch/eigen/eigen.hpp"

#include <algorithm>

namespace numbirch {

template<numeric T>
NUMBIRCH_KEEP Array<int,0> count(const T& x) {
  if constexpr (is_arithmetic_v<T>) {
    return count_functor()(x);
  } else if (size(x) == 0) {
    return 0;
  } else {
    return std::count_if(x.begin(), x.end(), count_functor());
  }
}

template<numeric T>
NUMBIRCH_KEEP Array<value_t<T>,0> sum(const T& x) {
  if constexpr (is_scalar_v<T>) {
    return x;
  } else if (size(x) == 0) {
    return value_t<T>(0);
  } else {
    return std::reduce(x.begin(), x.end());
  }
}

template<numeric T>
NUMBIRCH_KEEP Array<value_t<T>,0> min(const T& x) {
  if constexpr (is_scalar_v<T>) {
    return x;
  } else if (size(x) == 0) {
    return value_t<T>(0);
  } else {
    return *std::min_element(x.begin(), x.end());
  }
}

template<numeric T>
NUMBIRCH_KEEP Array<value_t<T>,0> max(const T& x) {
  if constexpr (is_scalar_v<T>) {
    return x;
  } else if (size(x) == 0) {
    return value_t<T>(0);
  } else {
    return *std::max_element(x.begin(), x.end());
  }
}

template<numeric T>
NUMBIRCH_KEEP T cumsum(const T& x) {
  if constexpr (is_scalar_v<T>) {
    return x;
  } else if (size(x) == 0) {
    return x;
  } else {
    T y(shape(x));
    std::inclusive_scan(x.begin(), x.end(), y.begin());
    return y;
  }
}

}
