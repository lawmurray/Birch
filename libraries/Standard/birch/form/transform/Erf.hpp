/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct ErfOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::erf(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::erf_grad(std::forward<G>(g), birch::eval(x));
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
using Erf = Form<ErfOp,T>;

template<argument T>
auto erf(T&& x) {
  return Erf<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
