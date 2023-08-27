/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct DigammaOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::digamma(birch::eval(x));
  }

  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::digamma(birch::eval(x), birch::eval(y));
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
using Digamma = Form<DigammaOp,Args...>;

template<argument T>
auto digamma(T&& x) {
  return Digamma<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T, argument U>
auto digamma(T&& x, U&& y) {
  return Digamma<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

}
