/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct Chol {
  BIRCH_UNARY_FORM(Chol)
};

BIRCH_UNARY_SIZE(Chol)
BIRCH_UNARY(Chol, chol)
BIRCH_UNARY_GRAD(Chol, chol_grad)

}
