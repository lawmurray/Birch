/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct SimulateBinomialOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::simulate_binomial(birch::eval(x), birch::eval(y));
  }

  template<class T, class U>
  static int rows(const T& x, const U& y) {
    return birch::rows(x, y);
  }

  template<class T, class U>
  static int columns(const T& x, const U& y) {
    return birch::columns(x, y);
  }
};

template<argument T, argument U>
using SimulateBinomial = Form<SimulateBinomialOp,T,U>;

template<argument T, argument U>
auto simulate_binomial(T&& x, U&& y) {
  return SimulateBinomial<tag_t<T>,tag_t<U>>(std::in_place,
      std::forward<T>(x), std::forward<U>(y));
}

}
