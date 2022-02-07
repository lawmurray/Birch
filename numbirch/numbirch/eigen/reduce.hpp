/**
 * @file
 */
#pragma once

#include "numbirch/reduce.hpp"
#include "numbirch/eigen/eigen.hpp"
#include "numbirch/eigen/transform.hpp"
#include "numbirch/common/functor.hpp"
#include "numbirch/common/get.hpp"

namespace numbirch {

template<class T, class>
Array<int,0> count(const T& x) {
  return make_eigen(x).unaryExpr(count_functor()).sum();
}

template<class T, class>
Array<value_t<T>,0> sum(const T& x) {
  return make_eigen(x).sum();
}

}
