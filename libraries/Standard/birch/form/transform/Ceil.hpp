/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct CeilOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::ceil(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::ceil_grad(std::forward<G>(g), birch::eval(x));
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
using Ceil = Form<CeilOp,T>;

template<argument T>
auto ceil(T&& x) {
  return Ceil<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
auto ceil(const Ceil<T>& x) {
  return x;
}

}
