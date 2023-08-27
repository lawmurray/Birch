/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct IotaOp {
  template<class T>
  static auto eval(const T& x, const int n) {
    return numbirch::iota(birch::eval(x), n);
  }

  template<class G, class T>
  static auto grad1(G&& g, const T& x, const int n) {
    return numbirch::iota_grad(std::forward<G>(g), birch::eval(x), n);
  }

  template<class T>
  static int rows(const T& x, const int n) {
    return n;
  }

  template<class T>
  static constexpr int columns(const T& x, const int n) {
    return 1;
  }
};

template<argument T>
using Iota = Form<IotaOp,T,int>;

template<argument T>
auto iota(T&& x, const int n) {
  return Iota<tag_t<T>>(std::in_place, std::forward<T>(x), n);
}

}
