/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct AsinOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::asin(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::asin_grad(std::forward<G>(g), birch::eval(x));
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
using Asin = Form<AsinOp,T>;

template<argument T>
auto asin(T&& x) {
  return Asin<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
