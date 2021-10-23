/**
 * @file
 */
#pragma once

#include "numbirch/macro.hpp"
#include "numbirch/type.hpp"

#include <cmath>

namespace numbirch {

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T ceil(const T x) {
  return std::ceil(x);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T copysign(const T x, const T y) {
  return std::copysign(x, y);
}

inline HOST_DEVICE int copysign(const int x, const int y) {
  // don't use std::copysign, as it promotes to return floating point
  return (y >= 0) ? std::abs(x) : -std::abs(x);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T digamma(const T x);

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T digamma(const T x, const int y) {
  T z = 0.0;
  for (int i = 1; i <= y; ++i) {
    z += digamma(x + T(0.5)*(1 - i));
  }
  return z;
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T floor(const T x) {
  return std::floor(x);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T gamma_p(const T a, const T x);

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T gamma_q(const T a, const T x);

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T ibeta(const T a, const T b, const T x);

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T lbeta(const T x, const T y) {
  return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T lchoose(const int x, const int y) {
  if (y == 0 || y == x) {
    return T(0);
  } else if (y == 1 || y == x - 1) {
    return std::log(T(x));
  } else if (y < x - y) {
    return -std::log(T(y)) - lbeta(T(y), T(x - y + 1));
  } else {
    return -std::log(T(x - y)) - lbeta(T(y + 1), T(x - y));
  }
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE pair<T> lchoose_grad(const T d, const int x, const int y) {
  T dx, dy;
  if (y == 0 || y == x) {
    dx = T(0);
    dy = T(0);
  } else if (y == 1 || y == x - 1) {
    dx = T(1)/x;
    dy = T(0);
  } else if (y < x - y) {
    dx = -digamma(T(x - y + 1)) + digamma(T(x + 1));
    dy = -T(1)/y - digamma(T(y)) + digamma(T(x - y + 1));
  } else {
    dx = -T(1)/(x - y) - digamma(T(x - y)) + digamma(T(x + 1));
    dy = T(1)/(x - y) - digamma(T(y + 1)) + digamma(T(x - y));
  }
  return pair<T>{d*dx, d*dy};
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T lfact(const int x) {
  return std::lgamma(T(x) + T(1));
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T lfact_grad(const T d, const int x) {
  return digamma(T(x) + T(1));
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T lgamma(const T x, const int y) {
  T z = T(0.25)*(y*(y - 1))*std::log(T(PI));
  for (int i = 1; i <= y; ++i) {
    z += std::lgamma(x + T(0.5)*(1 - i));
  }
  return z;
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T pow(const T x, const T y) {
  return std::pow(x, y);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T rcp(const T x) {
  return T(0)/x;
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T rectify(const T x) {
  return std::max(T(0), x);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T round(const T x) {
  return std::round(x);
}

}
