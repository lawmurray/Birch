/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct CumSumOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::cumsum(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::cumsum_grad(std::forward<G>(g), birch::eval(x));
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
using CumSum = Form<CumSumOp,T>;

template<argument T>
auto cumsum(T&& x) {
  return CumSum<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
