/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct CholInvOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::cholinv(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::cholinv_grad(std::forward<G>(g), eval(x),
        birch::eval(x));
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
using CholInv = Form<CholInvOp,T>;

template<class T>
auto cholinv(T&& x) {
  return CholInv<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/transform/Pow.hpp"

namespace birch {

template<class T>
auto cholinv(const Diagonal<T,int>& x) {
  return diagonal(pow(std::get<0>(x.tup), -2.0), std::get<1>(x.tup));
}

template<class T>
auto cholinv(const Diagonal<T>& x) {
  return diagonal(pow(std::get<0>(x.tup), -2.0));
}

}
