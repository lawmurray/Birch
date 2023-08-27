/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

template<numbirch::arithmetic To>
struct CastOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::cast<To>(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::cast_grad<To>(std::forward<G>(g), birch::eval(x));
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

template<numbirch::arithmetic To, argument T>
using Cast = Form<CastOp<To>,T>;

template<numbirch::arithmetic To, argument T>
auto cast(T&& x) {
  return Cast<To,tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<numbirch::arithmetic To, argument T>
auto cast(const Cast<To,T>& x) {
  return x;
}

}
