/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct LessOrEqualOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::less_or_equal(birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::less_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::less_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
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

template<argument T, argument U>
using LessOrEqual = Form<LessOrEqualOp,T,U>;

template<argument T, argument U>
auto less_or_equal(T&& x, U&& y) {
  return LessOrEqual<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

template<argument T, argument U>
requires (!numbirch::arithmetic<T> || !numbirch::arithmetic<U>)
auto operator<=(T&& x, U&& y) {
  return less_or_equal(std::forward<T>(x), std::forward<U>(y));
}

}
