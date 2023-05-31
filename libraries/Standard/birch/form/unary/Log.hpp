/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Log {
  BIRCH_UNARY_FORM(Log, numbirch::log)
  BIRCH_UNARY_GRAD(numbirch::log_grad)
  BIRCH_FORM
};

template<class Middle>
auto log(const Middle& m) {
  if constexpr (numbirch::is_arithmetic_v<Middle>) {
    return numbirch::log(m);
  } else {
    return BIRCH_UNARY_CONSTRUCT(Log);
  }
}

}
