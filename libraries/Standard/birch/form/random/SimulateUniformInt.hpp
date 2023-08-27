/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct SimulateUniformIntOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::simulate_uniform_int(birch::eval(x), birch::eval(y));
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
using SimulateUniformInt = Form<SimulateUniformIntOp,T,U>;

template<argument T, argument U>
auto simulate_uniform_int(T&& x, U&& y) {
  return SimulateUniformInt<tag_t<T>,tag_t<U>>(std::in_place,
      std::forward<T>(x), std::forward<U>(y));
}

}
