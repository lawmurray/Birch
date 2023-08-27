/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct DotOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::dot(birch::eval(x));
  }

  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::dot(birch::eval(x), birch::eval(y));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::dot_grad(std::forward<G>(g), birch::eval(x));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::dot_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::dot_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class T>
  static constexpr int rows(const T& x) {
    return 1;
  }

  template<class T, class U>
  static constexpr int rows(const T& x, const U& y) {
    return 1;
  }

  template<class T>
  static constexpr int columns(const T& x) {
    return 1;
  }

  template<class T, class U>
  static constexpr int columns(const T& x, const U& y) {
    return 1;
  }
};

template<class... Args>
using Dot = Form<DotOp,Args...>;

template<argument T>
auto dot(T&& x) {
  return Dot<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T, argument U>
auto dot(T&& x, U&& y) {
  return Dot<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

}
