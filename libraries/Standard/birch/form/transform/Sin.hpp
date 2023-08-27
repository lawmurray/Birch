/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct SinOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::sin(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::sin_grad(std::forward<G>(g), birch::eval(x));
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
using Sin = Form<SinOp,T>;

template<argument T>
auto sin(T&& x) {
  return Sin<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
