/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct CholSolveOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::cholsolve(birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::cholsolve_grad1(std::forward<G>(g), eval(x, y),
        birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::cholsolve_grad2(std::forward<G>(g), eval(x, y),
        birch::eval(x), birch::eval(y));
  }

  template<class T, class U>
  static int rows(const T& x, const U& y) {
    if constexpr (numbirch::scalar<decltype(birch::eval(y))>) {
      return birch::rows(x);
    } else {
      return birch::rows(y);
    }
  }

  template<class T, class U>
  static int columns(const T& x, const U& y) {
    if constexpr (numbirch::scalar<decltype(birch::eval(y))>) {
      return birch::columns(x);
    } else {
      return birch::columns(y);
    }
  }
};

template<argument T, argument U>
using CholSolve = Form<CholSolveOp,T,U>;

template<argument T, argument U>
auto cholsolve(T&& x, U&& y) {
  return CholSolve<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/transform/Pow.hpp"

namespace birch {

template<argument T, argument U>
auto cholsolve(const Diagonal<T,int>& x, U&& y) {
  return y/pow(std::get<0>(x.tup), 2.0);
}

}
