/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct Log1pOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::log1p(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::log1p_grad(std::forward<G>(g), birch::eval(x));
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
using Log1p = Form<Log1pOp,T>;

template<argument T>
auto log1p(T&& x) {
  return Log1p<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}

#include "birch/form/transform/Expm1.hpp"

namespace birch {

template<argument T>
decltype(auto) log1p(const Expm1<T>& x) {
  return std::get<0>(x.tup);
}

}
