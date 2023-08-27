/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct CholOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::chol(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::chol_grad(std::forward<G>(g), eval(x), birch::eval(x));
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

template<class T>
using Chol = Form<CholOp,T>;

template<class T>
auto chol(T&& x) {
  return Chol<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/transform/Sqrt.hpp"

namespace birch {

template<class T>
auto chol(const Diagonal<T,int>& x) {
  return diagonal(sqrt(std::get<0>(x.tup)), std::get<1>(x.tup));
}

template<class T>
auto chol(const Diagonal<T>& x) {
  return diagonal(sqrt(std::get<0>(x.tup)));
}

}
