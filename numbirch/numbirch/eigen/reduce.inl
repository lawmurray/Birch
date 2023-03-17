/**
 * @file
 */
#pragma once

#include "numbirch/reduce.hpp"
#include "numbirch/eigen/eigen.hpp"

namespace numbirch {

template<class T, class>
Array<int,0> count(const T& x) {
  if constexpr (is_arithmetic_v<T>) {
    return count_functor()(x);
  } else if (size(x) == 0) {
    return 0;
  } else {
    return make_eigen(x).unaryExpr(count_functor()).sum();
  }
}

template<class T, class>
Array<value_t<T>,0> sum(const T& x) {
  if constexpr (is_scalar_v<T>) {
    return x;
  } else if (size(x) == 0) {
    return value_t<T>(0);
  } else {
    return make_eigen(x).sum();
  }
}

}
