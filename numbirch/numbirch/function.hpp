/**
 * @file
 */
#pragma once

#include "numbirch/macro.hpp"

#include <boost/math/special_functions/digamma.hpp>
#include <cmath>

namespace numbirch {

template<class T, std::enable_if_t<!std::is_integral<T>::value,int> = 0>
HOST_DEVICE auto ceil(const T x) {
  return std::ceil(x);
}

template<class T, std::enable_if_t<std::is_integral<T>::value,int> = 0>
HOST_DEVICE auto ceil(const T x) {
  return x;
}

template<class T, class U, std::enable_if_t<!std::is_integral<T>::value ||
    !std::is_integral<U>::value,int> = 0>
HOST_DEVICE auto copysign(const T x, const U y) {
  return std::copysign(x, y);
}

template<class T, class U, std::enable_if_t<std::is_integral<T>::value &&
    std::is_integral<U>::value,int> = 0>
HOST_DEVICE auto copysign(const T x, const U y) {
  // std::copysign returns floating point here, override to return int
  return (y >= 0) ? std::abs(x) : -std::abs(x);
}

template<class T>
HOST_DEVICE auto digamma(const T x) {
  return boost::math::digamma(x);
}

template<class T>
HOST_DEVICE auto digamma(const T x, const int y) {
  T z = 0.0;
  for (int i = 1; i <= y; ++i) {
    z += digamma(x + T(0.5)*(1 - i));
  }
  return z;
}

template<class T, std::enable_if_t<!std::is_integral<T>::value,int> = 0>
HOST_DEVICE auto floor(const T x) {
  return std::floor(x);
}

template<class T, std::enable_if_t<std::is_integral<T>::value,int> = 0>
HOST_DEVICE auto floor(const T x) {
  return x;
}

template<class T, class U>
HOST_DEVICE auto lbeta(const T x, const U y) {
  return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
}

template<class T, class U>
HOST_DEVICE auto lchoose(const T x, const U y) {
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
HOST_DEVICE auto lgamma(const T x, const U y) {
  T z = T(0.25)*(y*(y - U(1)))*std::log(PI);
  for (U i = 1; i <= y; ++i) {
    z += std::lgamma(x + T(0.5)*(U(1) - i));
  }
  return z;
}

template<class T, class U, std::enable_if_t<!std::is_integral<T>::value ||
    !std::is_integral<U>::value,int> = 0>
HOST_DEVICE auto pow(const T x, const U y) {
  return std::pow(x, y);
}

template<class T, class U, std::enable_if_t<std::is_integral<T>::value &&
    std::is_integral<U>::value,int> = 0>
HOST_DEVICE auto pow(const T x, const U y) {
  // std::pow returns floating point here, override to return int
  return decltype(x*y)(std::pow(x, y));
}

template<class T>
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
