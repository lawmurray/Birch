/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct SimulateExponentialOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::simulate_exponential(birch::eval(x));
  }

  template<class T>
  static int rows(const T& x) {
    return birch::rows(x);
  }

  template<class T>
  static int columns(const T& x) {
    return birch::columns(x);
  }
};

template<argument T>
using SimulateExponential = Form<SimulateExponentialOp,T>;

template<argument T>
auto simulate_exponential(T&& x) {
  return SimulateExponential<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
