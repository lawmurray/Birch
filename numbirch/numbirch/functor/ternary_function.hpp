/**
 * @file
 */
#pragma once

#include "numbirch/functor/macro.hpp"
#include "numbirch/functor/function.hpp"

namespace numbirch {

template<class T, class U>
struct if_then_else_functor {
  DEVICE T operator()(const T x, const U y, const U z) const {
    return x ? y : z;
  }
};

}
