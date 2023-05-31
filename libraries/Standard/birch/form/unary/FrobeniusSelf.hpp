/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct FrobeniusSelf {
  BIRCH_UNARY_FORM(FrobeniusSelf, numbirch::frobenius)
  BIRCH_UNARY_GRAD(numbirch::frobenius_grad)
  BIRCH_FORM
};

template<class Middle>
auto frobenius(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(FrobeniusSelf);
}

}
