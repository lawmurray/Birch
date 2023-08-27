/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct ExpOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::exp(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::exp_grad(std::forward<G>(g), birch::eval(x));
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
using Exp = Form<ExpOp,T>;

template<argument T>
auto exp(T&& x) {
  return Exp<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}

#include "birch/form/transform/Log.hpp"

namespace birch {

template<argument T>
decltype(auto) exp(const Log<T>& x) {
  return std::get<0>(x.tup);
}

}
