/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct GreaterOrEqualOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::greater_or_equal(birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::greater_or_equal_grad1(std::forward<G>(g),
        birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::greater_or_equal_grad2(std::forward<G>(g),
        birch::eval(x), birch::eval(y));
  }

  template<class T, class U>
  static int rows(const T& x, const U& y) {
    return birch::rows(x, y);
  }

  template<class T, class U>
  static int columns(const T& x, const U& y) {
    return birch::columns(x, y);
  }
};

template<argument T, argument U>
using GreaterOrEqual = Form<GreaterOrEqualOp,T,U>;

template<argument T, argument U>
auto greater_or_equal(T&& x, U&& y) {
  return GreaterOrEqual<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

template<argument T, argument U>
requires (!numbirch::arithmetic<T> || !numbirch::arithmetic<U>)
auto operator>=(T&& x, U&& y) {
  return greater_or_equal(std::forward<T>(x), std::forward<U>(y));
}

}
