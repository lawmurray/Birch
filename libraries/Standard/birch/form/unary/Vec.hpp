/**
 * @file
 */
#pragma once

#include "birch/form/Unary.hpp"

namespace birch {

template<class Middle>
struct Vec {
  BIRCH_UNARY_FORM(Vec)
};

BIRCH_UNARY_SIZE(Vec)
BIRCH_UNARY(Vec, numbirch::vec)
BIRCH_UNARY_GRAD(Vec, numbirch::vec_grad)

template<class Middle>
auto vec(const Middle& m) {
  return BIRCH_UNARY_CONSTRUCT(Vec);
}

}
