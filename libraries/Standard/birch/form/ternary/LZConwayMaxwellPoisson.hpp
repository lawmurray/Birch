/**
 * @file
 */
#pragma once

#include "birch/form/Ternary.hpp"

namespace birch {

template<argument Left, argument Middle, argument Right>
struct LZConwayMaxwellPoisson {
  BIRCH_TERNARY_FORM(LZConwayMaxwellPoisson)
  BIRCH_TERNARY_SIZE(LZConwayMaxwellPoisson)
  BIRCH_TERNARY_EVAL(LZConwayMaxwellPoisson, lz_conway_maxwell_poisson)
  BIRCH_TERNARY_GRAD(LZConwayMaxwellPoisson, lz_conway_maxwell_poisson_grad)
};

BIRCH_TERNARY_TYPE(LZConwayMaxwellPoisson)
BIRCH_TERNARY_CALL(LZConwayMaxwellPoisson, lz_conway_maxwell_poisson)

}
