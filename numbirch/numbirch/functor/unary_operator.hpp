/**
 * @file
 */
#pragma once

#include "numbirch/functor/macro.hpp"
#include "numbirch/functor/function.hpp"

namespace numbirch {
struct logical_not_functor {
  DEVICE bool operator()(const bool x) const {
    return !x;
  }
};

template<class T>
struct negate_functor {
  DEVICE T operator()(const T x) const {
    return -x;
  }
};
}
