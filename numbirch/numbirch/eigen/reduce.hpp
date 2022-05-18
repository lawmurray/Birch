/**
 * @file
 */
#pragma once

#include "numbirch/eigen/eigen.hpp"
#include "numbirch/common/reduce.hpp"
#include "numbirch/reduce.hpp"

namespace numbirch {

template<class T, class>
Array<int,0> count(const T& x) {
  if constexpr (is_arithmetic_v<T>) {
    return count_functor()(x);
  } else {
    return make_eigen(x).unaryExpr(count_functor()).sum();
  }
}

template<class T, class>
Array<value_t<T>,0> sum(const T& x) {
  if constexpr (is_arithmetic_v<T>) {
    return x;
  } else {
    return make_eigen(x).sum();
  }
}

}
