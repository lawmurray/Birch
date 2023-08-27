/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct MinOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::min(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::min_grad(std::forward<G>(g), eval(x), birch::eval(x));
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
using Min = Form<MinOp,T>;

template<argument T>
auto min(T&& x) {
  return Min<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
