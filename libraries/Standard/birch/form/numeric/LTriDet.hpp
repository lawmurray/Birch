/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct LTriDetOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::ltridet(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::ltridet_grad(std::forward<G>(g), birch::eval(x));
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
using LTriDet = Form<LTriDetOp,T>;

template<argument T>
auto ltridet(T&& x) {
  return LTriDet<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}

#include "birch/form/array/Diagonal.hpp"
#include "birch/form/numeric/Mul.hpp"
#include "birch/form/transform/Log.hpp"
#include "birch/form/reduce/Sum.hpp"

namespace birch {

template<argument T>
auto ltridet(const Diagonal<T,int>& x) {
  return std::get<1>(x.tup)*log(std::get<0>(x.tup));
}

template<argument T>
auto ltridet(const Diagonal<T>& x) {
  return sum(log(std::get<0>(x.tup)));
}

}
