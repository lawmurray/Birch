/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct FloorOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::floor(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::floor_grad(std::forward<G>(g), birch::eval(x));
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

template<argument T>
using Floor = Form<FloorOp,T>;

template<argument T>
auto floor(T&& x) {
  return Floor<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
auto floor(const Floor<T>& x) {
  return x;
}

}
