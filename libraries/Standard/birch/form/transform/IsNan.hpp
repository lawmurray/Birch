/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct IsNanOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::isnan(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::isnan_grad(std::forward<G>(g), birch::eval(x));
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
using IsNan = Form<IsNanOp,T>;

template<argument T>
auto isnan(T&& x) {
  return IsNan<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
