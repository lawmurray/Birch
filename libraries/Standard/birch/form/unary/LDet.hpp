/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct LDet {
  BIRCH_UNARY_FORM(LDet)
};

BIRCH_UNARY_SIZE(LDet)
BIRCH_UNARY(LDet, numbirch::ldet)
BIRCH_UNARY_GRAD(LDet, numbirch::ldet_grad)

template<class Middle>
auto ldet(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(LDet);
}

}
