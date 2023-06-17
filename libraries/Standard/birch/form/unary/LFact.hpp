/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<argument Middle>
struct LFact {
  BIRCH_UNARY_FORM(LFact)
};

BIRCH_UNARY_SIZE(LFact)
BIRCH_UNARY(LFact, lfact)
BIRCH_UNARY_GRAD(LFact, lfact_grad)

}
