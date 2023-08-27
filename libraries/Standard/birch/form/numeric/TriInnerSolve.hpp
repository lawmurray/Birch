/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct TriInnerSolveOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::triinnersolve(birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::triinnersolve_grad1(std::forward<G>(g), eval(x, y),
        birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::triinnersolve_grad2(std::forward<G>(g), eval(x, y),
        birch::eval(x), birch::eval(y));
  }

  template<class T, class U>
  static int rows(const T& x, const U& y) {
    if constexpr (numbirch::scalar<decltype(birch::eval(y))>) {
      return birch::columns(x);
    } else {
      return birch::rows(y);
    }
  }

  template<class T, class U>
  static int columns(const T& x, const U& y) {
    if constexpr (numbirch::scalar<decltype(birch::eval(y))>) {
      return birch::rows(x);
    } else {
      return birch::columns(y);
    }
  }
};

template<class T, class U>
using TriInnerSolve = Form<TriInnerSolveOp,T,U>;

template<class T, class U>
auto triinnersolve(T&& x, U&& y) {
  return TriInnerSolve<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

}
