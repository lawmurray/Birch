/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct LCholDet {
  BIRCH_UNARY_FORM(LCholDet, numbirch::lcholdet)
  BIRCH_UNARY_GRAD(numbirch::lcholdet_grad)
  BIRCH_FORM
};

template<class Middle>
auto lcholdet(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(LCholDet);
}

}
