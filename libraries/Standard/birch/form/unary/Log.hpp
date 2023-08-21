/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct LogOp {
  template<class... Args>
  static auto eval(const Args&... args) {
    return numbirch::log(birch::eval(args)...);
  }

  template<class G, class... Args>
  static auto grad1(G&& g, const Args&... args) {
    return numbirch::log_grad(std::forward<G>(g), birch::eval(args)...);
  }

  template<class... Args>
  static int rows(const Args&... args) {
    return birch::rows(args...);
  }

  template<class... Args>
  static int columns(const Args&... args) {
    return birch::columns(args...);
  }
};

template<class T>
using Log = Form<LogOp,T>;

template<argument T>
auto log(T&& x) {
  return Log<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
decltype(auto) log(const Exp<T>& x) {
  return std::get<0>(x.tup);
}

}
