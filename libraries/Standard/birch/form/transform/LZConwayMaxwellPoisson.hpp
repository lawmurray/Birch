/**
 * @file
 */
#pragma once

#include "birch/form/Form.hpp"

namespace birch {

struct LZConwayMaxwellPoissonOp {
  template<class T, class U, class V>
  static auto eval(const T& x, const U& y, const V& z) {
    return numbirch::lz_conway_maxwell_poisson(birch::eval(x), birch::eval(y),
        birch::eval(z));
  }

  template<class G, class T, class U, class V>
  static auto grad1(G&& g, const T& x, const U& y, const V& z) {
    return numbirch::lz_conway_maxwell_poisson_grad1(std::forward<G>(g),
        birch::eval(x), birch::eval(y), birch::eval(z));
  }

  template<class G, class T, class U, class V>
  static auto grad2(G&& g, const T& x, const U& y, const V& z) {
    return numbirch::lz_conway_maxwell_poisson_grad2(std::forward<G>(g),
        birch::eval(x), birch::eval(y), birch::eval(z));
  }

  template<class G, class T, class U, class V>
  static auto grad3(G&& g, const T& x, const U& y, const V& z) {
    return numbirch::lz_conway_maxwell_poisson_grad3(std::forward<G>(g),
        birch::eval(x), birch::eval(y), birch::eval(z));
  }

  template<class T, class U, class V>
  static int rows(const T& x, const U& y, const V& z) {
    return birch::rows(x, y, z);
  }

  template<class T, class U, class V>
  static int columns(const T& x, const U& y, const V& z) {
    return birch::columns(x, y, z);
  }
};

template<argument T, argument U, argument V>
using LZConwayMaxwellPoisson = Form<LZConwayMaxwellPoissonOp,T,U,V>;

template<argument T, argument U, argument V>
auto lz_conway_maxwell_poisson(T&& x, U&& y, V&& z) {
  return LZConwayMaxwellPoisson<tag_t<T>,tag_t<U>,tag_t<V>>(std::in_place,
      std::forward<T>(x), std::forward<U>(y), std::forward<V>(z));
}

}
