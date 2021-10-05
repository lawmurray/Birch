/**
 * @file
 */
#pragma once

#include "numbirch/functor/macro.hpp"
#include "numbirch/functor/function.hpp"

namespace numbirch {

template<class T>
struct copysign_functor {
  DEVICE T operator()(const T x, const T y) const {
    return std::copysign(x, y);
  }
};

template<class T, class U>
struct digamma_p_functor {
  DEVICE T operator()(const T x, const U y) const {
    T z = 0.0;
    for (U i = 1; i <= y; ++i) {
      z += digamma(x + T(0.5)*(U(1) - i));
    }
    return z;
  }
};

template<class T, class U>
struct lbeta_functor {
  DEVICE T operator()(const T x, const U y) const {
    return lbeta(x, y);
  }
};

template<class T, class U>
struct lchoose_functor {
  DEVICE T operator()(const T x, const U y) const {
    return lchoose(x, y);
  }
};

template<class T, class U>
struct lgammap_functor {
  DEVICE T operator()(const T x, const U y) const {
    return lgamma(x, y);
  }
};

template<class T, class U>
struct pow_functor {
  DEVICE T operator()(const T x, const U y) const {
    return std::pow(x, y);
  }
};

}
