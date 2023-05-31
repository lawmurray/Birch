/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct LTriDet {
  BIRCH_UNARY_FORM(LTriDet, numbirch::ltridet)
  BIRCH_UNARY_GRAD(numbirch::ltridet_grad)
  BIRCH_FORM
};

template<class Middle>
auto ltridet(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(LTriDet);
}

}
