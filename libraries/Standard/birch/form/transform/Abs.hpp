/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct AbsOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::abs(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::abs_grad(std::forward<G>(g), birch::eval(x));
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
using Abs = Form<AbsOp,T>;

template<argument T>
auto abs(T&& x) {
  return Abs<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
auto abs(const Abs<T>& x) {
  return x;
}

}
