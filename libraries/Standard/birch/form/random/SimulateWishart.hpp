/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct SimulateWishartOp {
  template<class T>
  static auto eval(const T& x, const int n) {
    return numbirch::simulate_wishart(birch::eval(x), n);
  }

  template<class T>
  static int rows(const T& x, const int n) {
    return n;
  }

  template<class T>
  static int columns(const T& x, const int n) {
    return n;
  }
};

template<argument T>
using SimulateWishart = Form<SimulateWishartOp,T,int>;

template<argument T>
auto simulate_wishart(T&& x, const int n) {
  return SimulateWishart<tag_t<T>>(std::in_place, std::forward<T>(x), n);
}

}
