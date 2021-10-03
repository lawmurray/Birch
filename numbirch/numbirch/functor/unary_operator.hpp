/**
 * @file
 */
#pragma once

#include "numbirch/functor/macro.hpp"

namespace numbirch {
template<class T>
struct negate_functor {
  HOST_DEVICE T operator()(const T x) const {
    return -x;
  }
};
}
