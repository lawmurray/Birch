/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct FillOp {
  template<class T>
  static auto eval(const T& x, const int n) {
    return numbirch::fill(birch::eval(x), n);
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x, const int n) {
    return numbirch::fill_grad(std::forward<G>(g), birch::eval(x), n);
  }

  template<class T>
  static int rows(const T& x, const int n) {
    return n;
  }

  template<class T>
  static constexpr int columns(const T& x, const int n) {
    return 1;
  }

  template<class T>
  static auto eval(const T& x, const int m, const int n) {
    return numbirch::fill(birch::eval(x), m, n);
  }

  template<class G, class T>
  static auto grad(G&& g, const T& x, const int m, const int n) {
    return numbirch::fill_grad(std::forward<G>(g), birch::eval(x), m, n);
  }

  template<class T>
  static int rows(const T& x, const int m, const int n) {
    return m;
  }

  template<class T>
  static int columns(const T& x, const int m, const int n) {
    return n;
  }
};

template<argument... Args>
using Fill = Form<FillOp,Args...>;

template<argument T>
auto fill(T&& x, const int n) {
  return Fill<tag_t<T>,int>(std::in_place, std::forward<T>(x), n);
}

template<argument T>
auto fill(T&& x, const int m, const int n) {
  return Fill<tag_t<T>,int,int>(std::in_place, std::forward<T>(x), m, n);
}

}
