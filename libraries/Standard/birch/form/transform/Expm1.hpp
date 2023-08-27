/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct Expm1Op {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::expm1(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::expm1_grad(std::forward<G>(g), birch::eval(x));
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
using Expm1 = Form<Expm1Op,T>;

template<argument T>
auto expm1(T&& x) {
  return Expm1<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}

#include "birch/form/transform/Log1p.hpp"

namespace birch {

template<argument T>
decltype(auto) expm1(const Log1p<T>& x) {
  return std::get<0>(x.tup);
}

}
