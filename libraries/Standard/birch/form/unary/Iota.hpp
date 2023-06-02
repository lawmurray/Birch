/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Iota {
  BIRCH_UNARY_FORM(Iota, n)
};

BIRCH_UNARY_SIZE(Iota)
BIRCH_UNARY(Iota, numbirch::iota, n)
BIRCH_UNARY_GRAD(Iota, numbirch::iota_grad, n)

template<class Middle>
auto iota(const Middle& m, const int n) {
  return BIRCH_UNARY_CONSTRUCT(Iota, n);
}

}
