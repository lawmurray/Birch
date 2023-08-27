/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct VecOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::vec(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::vec_grad(std::forward<G>(g), birch::eval(x));
  }

  template<class T>
  static int rows(const T& x) {
    return birch::size(x);
  }

  template<class T>
  static constexpr int columns(const T& x) {
    return 1;
  }
};

template<argument T>
using Vec = Form<VecOp,T>;

template<argument T>
auto vec(T&& x) {
  return Vec<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
