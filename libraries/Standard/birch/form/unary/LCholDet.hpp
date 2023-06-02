/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct LCholDet {
  BIRCH_UNARY_FORM(LCholDet)
};

BIRCH_UNARY_SIZE(LCholDet)
BIRCH_UNARY(LCholDet, numbirch::lcholdet)
BIRCH_UNARY_GRAD(LCholDet, numbirch::lcholdet_grad)

template<class Middle>
auto lcholdet(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(LCholDet);
}

}
