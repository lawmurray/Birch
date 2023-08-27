/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct TransposeOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::transpose(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::transpose_grad(std::forward<G>(g), birch::eval(x));
  }

  template<class T>
  static int rows(const T& x) {
    return birch::columns(x);
  }

  template<class T>
  static int columns(const T& x) {
    return birch::rows(x);
  }
};

template<argument T>
using Transpose = Form<TransposeOp,T>;

template<argument T>
auto transpose(T&& x) {
  return Transpose<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
decltype(auto) transpose(const Transpose<T>& x) {
  return std::get<0>(x.tup);
}

}
