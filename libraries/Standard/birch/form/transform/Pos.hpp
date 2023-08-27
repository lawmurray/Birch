/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct PosOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::pos(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::pos_grad(std::forward<G>(g), birch::eval(x));
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
using Pos = Form<PosOp,T>;

template<argument T>
auto pos(T&& x) {
  return Pos<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
auto pos(const Pos<T>& x) {
  return x;
}

template<argument T>
requires (!numbirch::arithmetic<T>)
auto operator+(T&& x) {
  return pos(std::forward<T>(x));
}

}
