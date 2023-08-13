/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Chol {
  BIRCH_UNARY_FORM(Chol)
  BIRCH_UNARY_SIZE(Chol)
  BIRCH_UNARY_EVAL(Chol, chol)
  BIRCH_UNARY_GRAD_WITH_RESULT(Chol, chol_grad)
};

BIRCH_UNARY_TYPE(Chol)
BIRCH_UNARY_CALL(Chol, chol)

}
