/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct LDet {
  BIRCH_UNARY_FORM(LDet, numbirch::ldet)
  BIRCH_UNARY_GRAD(numbirch::ldet_grad)
  BIRCH_FORM
};

template<class Middle>
auto ldet(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(LDet);
}

}
