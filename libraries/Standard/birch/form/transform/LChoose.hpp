/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct LChooseOp {
  template<class T, class U>
  static auto eval(const T& x, const U& y) {
    return numbirch::lchoose(birch::eval(x), birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad1(G&& g, const T& x, const U& y) {
    return numbirch::lchoose_grad1(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class G, class T, class U>
  static auto grad2(G&& g, const T& x, const U& y) {
    return numbirch::lchoose_grad2(std::forward<G>(g), birch::eval(x),
        birch::eval(y));
  }

  template<class T, class U>
  static int rows(const T& x, const U& y) {
    return birch::rows(x, y);
  }

  template<class T, class U>
  static int columns(const T& x, const U& y) {
    return birch::columns(x, y);
  }
};

template<argument T, argument U>
using LChoose = Form<LChooseOp,T,U>;

template<argument T, argument U>
auto lchoose(T&& x, U&& y) {
  return LChoose<tag_t<T>,tag_t<U>>(std::in_place, std::forward<T>(x),
      std::forward<U>(y));
}

}
