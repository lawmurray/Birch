/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct TriInv {
  BIRCH_UNARY_FORM(TriInv)
};

BIRCH_UNARY_SIZE(TriInv)
BIRCH_UNARY(TriInv, triinv)
BIRCH_UNARY_GRAD(TriInv, triinv_grad)

}
