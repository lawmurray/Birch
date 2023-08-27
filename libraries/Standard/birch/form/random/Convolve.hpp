/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct ConvolveOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::convolve(birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::convolve_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::convolve_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class T, class U>
  static int rows(const T& x, const U& y) {
    return birch::rows(x) + birch::rows(y) - 1;
  }

  template<class T, class U>
  static constexpr int columns(const T& x, const U& y) {
    return 1;
  }
};

template<argument T, argument U>
using Convolve = Form<ConvolveOp,T,U>;

template<argument T, argument U>
auto convolve(T&& x, U&& y) {
  return Convolve<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

}
