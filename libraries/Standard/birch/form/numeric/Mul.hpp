/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct MulOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::mul(birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::mul_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::mul_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class T, class U>
  static int rows(const T& x, const U& y) {
    if constexpr (numbirch::scalar<decltype(birch::eval(x))> ||
        numbirch::scalar<decltype(birch::eval(y))>) {
      return birch::rows(x, y);
    } else {
      return birch::rows(x);
    }
  }

  template<class T, class U>
  static int columns(const T& x, const U& y) {
    if constexpr (numbirch::scalar<decltype(birch::eval(x))> ||
        numbirch::scalar<decltype(birch::eval(y))>) {
      return birch::columns(x, y);
    } else {
      return birch::columns(y);
    }
  }
};

template<argument T, argument U>
using Mul = Form<MulOp,T,U>;

template<argument T, argument U>
auto mul(T&& x, U&& y) {
  return Mul<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

template<argument T, argument U>
requires (!numbirch::arithmetic<T> || !numbirch::arithmetic<U>)
auto operator*(T&& x, U&& y) {
  return mul(std::forward<T>(x), std::forward<U>(y));
}

}

#include "birch/form/array/Diagonal.hpp"

namespace birch {

template<argument T, argument U>
auto operator*(const Diagonal<T,int>& x, U&& y) {
  return diagonal(std::get<0>(x.tup)*y, std::get<1>(x.tup));
}

template<argument T, argument U>
auto operator*(T&& x, const Diagonal<U,int>& y) {
  return diagonal(x*std::get<0>(y.tup), std::get<1>(y.tup));
}

template<argument T, argument U>
auto operator*(const Diagonal<T,int>& x, const Diagonal<U,int>& y) {
  assert(std::get<1>(x.tup) == std::get<1>(y.tup));
  return diagonal(std::get<0>(x.tup)*std::get<0>(y.tup),
      std::get<1>(x.tup));
}

}
