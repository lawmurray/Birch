/**
 * @file
 */
#pragma once

#include "numbirch/functor/macro.hpp"

#include <boost/math/special_functions/digamma.hpp>
#include <cmath>

namespace numbirch {

template<class T, class U>
DEVICE T lbeta(const T x, const U y) {
  return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
}

template<class T, class U>
DEVICE T lchoose(const T x, const U y) {
  // based on the Boost binomial_coefficient implementation
  if (y == U(0) || y == x) {
    return T(0);
  } else if (y == U(1) || y == x - T(1)) {
    return std::log(x);
  } else if (y < x - y) {
    return -std::log(y) - lbeta(y, x - y + T(1));
  } else {
    return -std::log(x - y) - lbeta(y + T(1), x - y);
  }
}

template<class T, class U>
DEVICE T lgamma(const T x, const U y) {
  T z = T(0.25)*(y*(y - U(1)))*std::log(PI);
  for (U i = 1; i <= y; ++i) {
    z += std::lgamma(x + T(0.5)*(U(1) - i));
  }
  return z;
}

template<class T>
DEVICE T digamma(const T x) {
  return boost::math::digamma(x);
}

}
