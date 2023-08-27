/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct TriInvOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::triinv(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::triinv_grad(std::forward<G>(g), eval(x),
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
using TriInv = Form<TriInvOp,T>;

template<class T>
auto triinv(T&& x) {
  return TriInv<tag_t<T>>(std::in_place, std::forward<T>(x));
}

template<argument T>
auto triinv(const TriInv<T>& x) {
  return std::get<0>(x.tup);
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/transform/Div.hpp"

namespace birch {

template<class T>
auto triinv(const Diagonal<T,int>& x) {
  return diagonal(1.0/std::get<0>(x.tup), std::get<1>(x.tup));
}

template<class T>
auto triinv(const Diagonal<T>& x) {
  return diagonal(1.0/std::get<0>(x.tup));
}

}
