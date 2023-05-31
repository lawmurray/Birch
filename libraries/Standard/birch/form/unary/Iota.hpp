/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Iota {
  BIRCH_UNARY_FORM(Iota, numbirch::iota, n)
  BIRCH_UNARY_GRAD(numbirch::iota_grad, n)
  BIRCH_FORM
};

template<class Middle>
auto iota(const Middle& m, const int n) {
  return BIRCH_UNARY_CONSTRUCT(Iota, n);
}

}
