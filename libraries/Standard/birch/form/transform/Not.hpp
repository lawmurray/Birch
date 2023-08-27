/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct NotOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::neg(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::logical_not_grad(std::forward<G>(g), birch::eval(x));
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
using Not = Form<NotOp,T>;

template<argument T>
auto logical_not(T&& x) {
  return Not<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
auto logical_not(const Not<T>& x) {
  return std::get<0>(x.tup);
}

template<argument T>
requires (!numbirch::arithmetic<T>)
auto operator!(T&& x) {
  return logical_not(std::forward<T>(x));
}

}
