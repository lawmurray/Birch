/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct CumSum {
  BIRCH_UNARY_FORM(CumSum)
};

BIRCH_UNARY_SIZE(CumSum)
BIRCH_UNARY(CumSum, cumsum)
BIRCH_UNARY_GRAD(CumSum, cumsum_grad)

}
