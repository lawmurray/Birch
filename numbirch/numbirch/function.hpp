/**
 * @file
 */
#pragma once

#include "numbirch/macro.hpp"

#include <boost/math/special_functions/digamma.hpp>
#include <cmath>

namespace numbirch {

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE auto ceil(const T x) {
  return std::ceil(x);
}

template<class T, std::enable_if_t<std::is_integral<T>::value,int> = 0>
HOST_DEVICE auto ceil(const T x) {
  return x;
}

template<class T, class U, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value &&
    !(std::is_integral<T>::value && std::is_integral<U>::value),int> = 0>
HOST_DEVICE auto copysign(const T x, const U y) {
  return std::copysign(x, y);
}

template<class T, class U, std::enable_if_t<std::is_integral<T>::value &&
    std::is_integral<U>::value,int> = 0>
HOST_DEVICE auto copysign(const T x, const U y) {
  // std::copysign returns floating point here, override to return int
  return (y >= 0) ? std::abs(x) : -std::abs(x);
}

template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
HOST_DEVICE auto digamma(const T x) {
  return boost::math::digamma(x);
}

template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
HOST_DEVICE auto digamma(const T x, const int y) {
  using U = decltype(digamma(x));
  U z = 0.0;
  for (int i = 1; i <= y; ++i) {
    z += digamma(x + U(0.5)*(1 - i));
  }
  return z;
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE auto floor(const T x) {
  return std::floor(x);
}

template<class T, std::enable_if_t<std::is_integral<T>::value,int> = 0>
HOST_DEVICE auto floor(const T x) {
  return x;
}

template<class T, class U, std::enable_if_t<std::is_arithmetic<T>::value &&
    std::is_arithmetic<U>::value,int> = 0>
HOST_DEVICE auto lbeta(const T x, const U y) {
  return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
}

template<class T, class U, std::enable_if_t<std::is_arithmetic<T>::value &&
    std::is_arithmetic<U>::value,int> = 0>
HOST_DEVICE auto lchoose(const T x, const U y) {
  // based on the Boost binomial_coefficient implementation
  using V = decltype(std::log(x - y));
  if (y == 0 || y == x) {
    return V(0);
  } else if (y == 1 || y == x - 1) {
    return V(std::log(x));
  } else if (y < x - y) {
    return V(-std::log(y) - lbeta(y, x - y + 1));
  } else {
    return V(-std::log(x - y) - lbeta(y + 1, x - y));
  }
}

template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
HOST_DEVICE auto lgamma(const T x, const int y) {
  using U = decltype(std::lgamma(x));
  U z = U(0.25)*(y*(y - 1))*std::log(U(PI));
  for (U i = 1; i <= y; ++i) {
    z += std::lgamma(x + U(0.5)*(1 - i));
  }
  return z;
}

template<class T, class U, std::enable_if_t<
    std::is_arithmetic<T>::value && std::is_arithmetic<U>::value &&
    !(std::is_integral<T>::value && std::is_integral<U>::value),int> = 0>
HOST_DEVICE auto pow(const T x, const U y) {
  return std::pow(x, y);
}

template<class T, class U, std::enable_if_t<std::is_integral<T>::value &&
    std::is_integral<U>::value,int> = 0>
HOST_DEVICE auto pow(const T x, const U y) {
  // std::pow returns floating point here, override to return int
  return decltype(x*y)(std::pow(x, y));
}

template<class T, std::enable_if_t<std::is_arithmetic<T>::value,int> = 0>
HOST_DEVICE auto rectify(const T x) {
  return std::max(T(0), x);
}

template<class T, std::enable_if_t<!std::is_integral<T>::value,int> = 0>
HOST_DEVICE auto round(const T x) {
  return std::round(x);
}

template<class T, std::enable_if_t<std::is_integral<T>::value,int> = 0>
HOST_DEVICE auto round(const T x) {
  return x;
}

}
