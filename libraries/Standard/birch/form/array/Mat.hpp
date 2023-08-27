/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct MatOp {
  template<class T>
  static auto eval(const T& x, const int n) {
    return numbirch::mat(birch::eval(x), n);
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x, const int n) {
    return numbirch::mat_grad(std::forward<G>(g), birch::eval(x), n);
  }

  template<class T>
  static int rows(const T& x, const int n) {
    return birch::size(x)/n;
  }

  template<class T>
  static int columns(const T& x, const int n) {
    return n;
  }
};

template<argument T>
using Mat = Form<MatOp,T,int>;

template<argument T>
auto mat(T&& x, const int n) {
  return Mat<tag_t<T>>(std::in_place, std::forward<T>(x), n);
}

}
