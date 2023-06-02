/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct FrobeniusSelf {
  BIRCH_UNARY_FORM(FrobeniusSelf)
};

BIRCH_UNARY_SIZE(FrobeniusSelf)
BIRCH_UNARY(FrobeniusSelf, numbirch::frobenius)
BIRCH_UNARY_GRAD(FrobeniusSelf, numbirch::frobenius_grad)

template<class Middle>
auto frobenius(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(FrobeniusSelf);
}

}
