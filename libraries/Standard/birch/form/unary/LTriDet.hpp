/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct LTriDet {
  BIRCH_UNARY_FORM(LTriDet)
};

BIRCH_UNARY_SIZE(LTriDet)
BIRCH_UNARY(LTriDet, numbirch::ltridet)
BIRCH_UNARY_GRAD(LTriDet, numbirch::ltridet_grad)

template<class Middle>
auto ltridet(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(LTriDet);
}

}
