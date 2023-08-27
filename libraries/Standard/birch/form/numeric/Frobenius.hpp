/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct FrobeniusOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::frobenius(birch::eval(x));
  }

  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::frobenius(birch::eval(x), birch::eval(y));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::frobenius_grad(std::forward<G>(g), birch::eval(x));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::frobenius_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::frobenius_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class T>
  static constexpr int rows(const T& x) {
    return 1;
  }

  template<class T, class U>
  static constexpr int rows(const T& x, const U& y) {
    return 1;
  }

  template<class T>
  static constexpr int columns(const T& x) {
    return 1;
  }

  template<class T, class U>
  static constexpr int columns(const T& x, const U& y) {
    return 1;
  }
};

template<class... Args>
using Frobenius = Form<FrobeniusOp,Args...>;

template<argument T>
auto frobenius(T&& x) {
  return Frobenius<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T, argument U>
auto frobenius(T&& x, U&& y) {
  return Frobenius<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

}
