/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct IsInfOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::isinf(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::isinf_grad(std::forward<G>(g), birch::eval(x));
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
using IsInf = Form<IsInfOp,T>;

template<argument T>
auto isinf(T&& x) {
  return IsInf<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
