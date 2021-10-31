/**
 * @file
 */
#pragma once

#include "numbirch/function.hpp"

namespace numbirch {
struct logical_not_functor {
  HOST DEVICE bool operator()(const bool x) const {
    return !x;
  }
};

template<class T>
struct negate_functor {
  HOST DEVICE auto operator()(const T x) const {
    return -x;
  }
};
}
