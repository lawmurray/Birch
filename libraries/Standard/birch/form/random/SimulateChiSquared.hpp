/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct SimulateChiSquaredOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::simulate_chi_squared(birch::eval(x));
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
using SimulateChiSquared = Form<SimulateChiSquaredOp,T>;

template<argument T>
auto simulate_chi_squared(T&& x) {
  return SimulateChiSquared<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
