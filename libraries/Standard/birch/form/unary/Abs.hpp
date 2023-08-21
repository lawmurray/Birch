/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct AbsOp {
  template<class... Args>
  static auto eval(const Args&... args) {
    return numbirch::abs(birch::eval(args)...);
  }

  template<class G, class... Args>
  static auto grad1(G&& g, const Args&... args) {
    return numbirch::abs_grad(std::forward<G>(g), birch::eval(args)...);
  }

  template<class... Args>
  static int rows(const Args&... args) {
    return birch::rows(args...);
  }

  template<class... Args>
  static int columns(const Args&... args) {
    return birch::columns(args...);
  }
};

template<class T>
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
