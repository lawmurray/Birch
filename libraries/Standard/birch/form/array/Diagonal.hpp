/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct DiagonalOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::diagonal(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::diagonal_grad(std::forward<G>(g), birch::eval(x));
  }

  template<class T>
  static int rows(const T& x) {
    return rows(x);
  }

  template<class T>
  static int columns(const T& x) {
    return rows(x);
  }

  template<class T>
  static auto eval(const T& x, const int n) {
    return numbirch::diagonal(birch::eval(x), n);
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x, const int n) {
    return numbirch::diagonal_grad(std::forward<G>(g), birch::eval(x), n);
  }

  template<class T>
  static int rows(const T& x, const int n) {
    return n;
  }

  template<class T>
  static int columns(const T& x, const int n) {
    return n;
  }
};

template<argument... Args>
using Diagonal = Form<DiagonalOp,Args...>;

template<argument T>
auto diagonal(T&& x) {
  return Diagonal<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
auto diagonal(T&& x, const int n) {
  return Diagonal<tag_t<T>,int>(std::in_place, std::forward<T>(x), n);
}

inline auto identity(const int n) {
  return Diagonal<Real,int>(std::in_place, Real(1.0), n);
}

}
