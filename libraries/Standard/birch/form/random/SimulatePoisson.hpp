/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct SimulatePoissonOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::simulate_poisson(birch::eval(x));
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
using SimulatePoisson = Form<SimulatePoissonOp,T>;

template<argument T>
auto simulate_poisson(T&& x) {
  return SimulatePoisson<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
