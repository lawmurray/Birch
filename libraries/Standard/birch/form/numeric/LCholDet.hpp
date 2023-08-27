/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct LCholDetOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::lcholdet(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::lcholdet_grad(std::forward<G>(g), birch::eval(x));
  }

  template<class T>
  static constexpr int rows(const T& x) {
    return 1;
  }

  template<class T>
  static constexpr int columns(const T& x) {
    return 1;
  }
};

template<class T>
using LCholDet = Form<LCholDetOp,T>;

template<argument T>
auto lcholdet(T&& x) {
  return LCholDet<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/numeric/Mul.hpp"
#include "birch/form/transform/Log.hpp"
#include "birch/form/reduce/Sum.hpp"

namespace birch {

template<argument T>
auto lcholdet(const Diagonal<T,int>& x) {
  return 2.0*std::get<1>(x.tup)*log(std::get<0>(x.tup));
}

template<argument T>
auto lcholdet(const Diagonal<T>& x) {
  return 2.0*sum(log(std::get<0>(x.tup)));
}

}
