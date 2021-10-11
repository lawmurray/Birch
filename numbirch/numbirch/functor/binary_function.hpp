/**
 * @file
 */
#pragma once

#include "numbirch/function.hpp"

#include <type_traits>

namespace numbirch {

template<class T, class U>
struct copysign_functor {
  HOST_DEVICE auto operator()(const T x, const U y) const {
    return copysign(x, y);
  }
};

template<class T>
struct digamma_p_functor {
  HOST_DEVICE auto operator()(const T x, const int y) const {
    return digamma(x, y);
  }
};

template<class T, class U>
struct lbeta_functor {
  HOST_DEVICE auto operator()(const T x, const U y) const {
    return lbeta(x, y);
  }
};

template<class T, class U>
struct lchoose_functor {
  HOST_DEVICE auto operator()(const T x, const U y) const {
    return lchoose(x, y);
  }
};

template<class T, class U>
struct lgammap_functor {
  HOST_DEVICE auto operator()(const T x, const U y) const {
    return lgamma(x, y);
  }
};

template<class T, class U>
struct pow_functor {
  HOST_DEVICE auto operator()(const T x, const U y) const {
    return pow(x, y);
  }
};

}
