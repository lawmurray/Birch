/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct LogOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::log(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::log_grad(std::forward<G>(g), birch::eval(x));
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
using Log = Form<LogOp,T>;

template<argument T>
auto log(T&& x) {
  return Log<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}

#include "birch/form/transform/Exp.hpp"

namespace birch {

template<argument T>
decltype(auto) log(const Exp<T>& x) {
  return std::get<0>(x.tup);
}

}
