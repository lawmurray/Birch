/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct FrobeniusSelf {
  BIRCH_UNARY_FORM(FrobeniusSelf)
  BIRCH_UNARY_SIZE(FrobeniusSelf)
  BIRCH_UNARY_EVAL(FrobeniusSelf, frobenius)
  BIRCH_UNARY_GRAD(FrobeniusSelf, frobenius_grad)
};

BIRCH_UNARY_TYPE(FrobeniusSelf)
BIRCH_UNARY_CALL(FrobeniusSelf, frobenius)

}
