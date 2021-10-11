/**
 * @file
 */
#pragma once

#include "numbirch/function.hpp"

namespace numbirch {

template<class T>
struct if_then_else_functor {
  HOST_DEVICE auto operator()(const bool x, const T y, const T z) const {
    return x ? y : z;
  }
};

}
