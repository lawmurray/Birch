/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct NegOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::neg(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::neg_grad(std::forward<G>(g), birch::eval(x));
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
using Neg = Form<NegOp,T>;

template<argument T>
auto neg(T&& x) {
  return Neg<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
auto neg(const Neg<T>& x) {
  return std::get<0>(x.tup);
}

template<argument T>
requires (!numbirch::arithmetic<T>)
auto operator-(T&& x) {
  return neg(std::forward<T>(x));
}

}
