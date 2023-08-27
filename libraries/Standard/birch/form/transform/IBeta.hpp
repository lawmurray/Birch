/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct IBetaOp {
  template<class T, class U, class V>
  static auto eval(const T& x, const U& y, const V& z) {
    return numbirch::ibeta(birch::eval(x), birch::eval(y), birch::eval(z));
  }

  template<class T, class U, class V>
  static int rows(const T& x, const U& y, const V& z) {
    return birch::rows(x, y, z);
  }

  template<class T, class U, class V>
  static int columns(const T& x, const U& y, const V& z) {
    return birch::columns(x, y, z);
  }
};

template<argument T, argument U, argument V>
using IBeta = Form<IBetaOp,T,U,V>;

template<argument T, argument U, argument V>
auto ibeta(T&& x, U&& y, V&& z) {
  return IBeta<tag_t<T>,tag_t<U>,tag_t<V>>(std::in_place,
      std::forward<T>(x), std::forward<U>(y), std::forward<V>(z));
}

}
