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

template<class T, class U, std::enable_if_t<
    std::is_floating_point<T>::value && std::is_arithmetic<U>::value,int> = 0>
HOST_DEVICE T ibeta(const U a, const U b, const T x);

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T lbeta(const T x, const T y) {
  return std::lgamma(x) + std::lgamma(y) - std::lgamma(x + y);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T lfact(const int x) {
  return std::lgamma(T(x + 1));
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T lfact_grad(const T d, const int x) {
  return d*digamma(T(x + 1));
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T lchoose(const int x, const int y) {
  return lfact<T>(x) - lfact<T>(y) - lfact<T>(x - y);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE pair<T> lchoose_grad(const T d, const int x, const int y) {
  T dx = lfact_grad<T>(d, x) - lfact_grad<T>(d, x - y);
  T dy = -lfact_grad<T>(d, y) + lfact_grad<T>(d, x - y);
  return pair<T>{dx, dy};
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T lgamma(const T x, const int y) {
  T z = T(0.25)*(y*T(y - 1))*std::log(T(PI));
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
HOST_DEVICE T rectify_grad(const T d, const T x) {
  return (x > 0) ? d : T(0);
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value,int> = 0>
HOST_DEVICE T round(const T x) {
  return std::round(x);
}

}
