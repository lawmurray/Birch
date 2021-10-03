/**
 * @file
 */
#pragma once

#include "numbirch/functor/macro.hpp"

namespace numbirch {
template<class T>
struct log_functor {
  HOST_DEVICE T operator()(const T x) const {
    return std::log(x);
  }
};

template<class T>
struct rectify_functor {
  HOST_DEVICE T operator()(const T x) const {
    return x > T(0) ? x : T(0);
  }
};
}
