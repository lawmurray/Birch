/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct MaxOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::max(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::max_grad(std::forward<G>(g), eval(x), birch::eval(x));
  }

  template<class T>
  static constexpr int rows(const T& x) {
    return 1;
  }

  template<class T>
  static constexpr int columns(const T& x) {
    return 1;
  }
};

template<argument T>
using Max = Form<MaxOp,T>;

template<argument T>
auto max(T&& x) {
  return Max<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
