/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct SimulateDirichletOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::simulate_dirichlet(birch::eval(x));
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
using SimulateDirichlet = Form<SimulateDirichletOp,T>;

template<argument T>
auto simulate_dirichlet(T&& x) {
  return SimulateDirichlet<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
