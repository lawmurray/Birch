/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct SumOp {
  template<class T>
  static auto eval(const T& x) {
    return numbirch::sum(birch::eval(x));
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x) {
    return numbirch::sum_grad(std::forward<G>(g), birch::eval(x));
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

template<argument T>
using Sum = Form<SumOp,T>;

template<argument T>
auto sum(T&& x) {
  return Sum<tag_t<T>>(std::in_place, std::forward<T>(x));
}

}
