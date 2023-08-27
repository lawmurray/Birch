/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct LGammaOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::lgamma(birch::eval(x));
  }

  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::lgamma(birch::eval(x), birch::eval(y));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::lgamma_grad(std::forward<G>(g), birch::eval(x));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::lgamma_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::lgamma_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class T>
  static int rows(const T& x) {
    return birch::rows(x);
  }

  template<class T, class U>
  static int rows(const T& x, const U& y) {
    return birch::rows(x, y);
  }

  template<class T>
  static int columns(const T& x) {
    return birch::columns(x);
  }

  template<class T, class U>
  static int columns(const T& x, const U& y) {
    return birch::columns(x, y);
  }
};

template<argument... Args>
using LGamma = Form<LGammaOp,Args...>;

template<argument T>
auto lgamma(T&& x) {
  return LGamma<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T, argument U>
auto lgamma(T&& x, U&& y) {
  return LGamma<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

}
